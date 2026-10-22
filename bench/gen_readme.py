#!/usr/bin/env python3
"""Writes the results table of bench/README.md from bench/results/**/*.json. Stdlib only, no GPU, no model.

  python3 bench/gen_readme.py            rewrite the table between the BEGIN/END markers of bench/README.md
  python3 bench/gen_readme.py --check    exit 1 if bench/README.md is not what this script writes, whoever the
                                         "Submitted by" names are (CI runs it)

One row per run record (schema 2, written by bench/harness/bench.py). JSON files that are not run records (a
model manifest, a machine.json, a list such as 2026-10-21-kflash-trees/kflash-merge.json) are skipped, the same way
tests/test_bench.py skips them. Everything but "Submitted by" comes from the record. "Submitted by" is the author of
the git commit that first added the file, so the script needs the full git history (a shallow clone is refused).
Missing numbers are shown as "–"; nothing is computed or filled in.
"""
import argparse, glob, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
RESULTS = os.path.join(HERE, "results")
README = os.path.join(HERE, "README.md")
BEGIN = "<!-- BEGIN GENERATED: results (python3 bench/gen_readme.py; do not edit by hand) -->"
END = "<!-- END GENERATED: results -->"
DASH = "–"

# the quantization, from the weights' path, then the run's name; first match wins
QUANTS = [
    (r"nvfp4", "NVFP4"), (r"\bfp8\b|-fp8", "FP8"), (r"bf16", "bf16"), (r"24gb", "2-bit experts"),
    (r"4\.7bpw", "4.7-bit"), (r"q4_k_m", "Q4_K_M"), (r"q5_k_m", "Q5_K_M"), (r"q4_0", "Q4_0"), (r"q4_1", "Q4_1"), (r"iq4_nl", "IQ4_NL"),
    (r"iq2_xxs|2bit|2-bit", "2-bit experts"), (r"oq4e", "4-bit"), (r"8bit", "8-bit"), (r"6bit", "6-bit"),
    (r"5bit", "5-bit"), (r"4bit|4-bit|mlx-serve|mtplx|omlx|rapid|\bmtp\b", "4-bit"),
]


def quant(r):
    for text in (str(r.get("model_path", "")), str(r.get("model_name", "")), r.get("label", ""), r.get("display", "")):
        t = text.lower()
        for pat, name in QUANTS:
            if re.search(pat, t):
                return name
    return DASH


def model(r):
    n = re.sub(r"\s*\((bf16|fp8|nvfp4)\)$", "", r["model_name"])
    n = re.sub(r"^Nemotron-Labs-", "Nemotron ", n)
    n = re.sub(r"^Nemotron-3\.5-", "Nemotron 3.5 ", n)
    n = re.sub(r"-(24GB|NVFP4)$", "", n)
    return n.replace("-", " ", 1) if n.startswith("Qwen") else n


def engine(r):
    return re.sub(r"\s*\([0-9a-f]{7}\)", "", r["display"])


def mac(r):
    s = r["provenance"]["system_start"]
    chip = re.sub(r"^Apple ", "", s.get("cpu", "?"))
    try:
        gb = f"{round(int(s['memsize']) / 2**30)} GB"
    except (KeyError, ValueError):
        gb = "?"
    return f"{chip}, {gb}"


def version(r):
    v, e = str(r.get("version", "")), r["engine"]
    if e in ("kapitsa", "nace-infer"):
        return v[:7] + ("+dirty" if v.endswith("+dirty") else "") if re.match(r"[0-9a-f]{7}", v) else (v.split() or [DASH])[0]
    m = re.search(r"\b(b\d{4,6})\b", v) if e == "llama.cpp" else None
    if m:
        return m.group(1)
    toks = v.split()
    if e == "omlx" and len(toks) > 1:
        return toks[1] if toks[0] == "omlx" else toks[0]
    return " ".join(toks[:2]) if toks[:1] and toks[0].startswith(("mlx", "mtplx")) else (toks[0] if toks else DASH)


def num(v, digits=0):
    return DASH if v is None else f"{v:.{digits}f}"


def cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def prefill(r, target):
    for p in r.get("prefill") or []:
        if p.get("target") == target:
            return p
    return {}


def row(path, r, who):
    dec = (r.get("decode_summary") or {}).get("median_tps")
    gre = (r.get("decode_greedy_summary") or {}).get("median_tps")
    p4, p16 = prefill(r, 4000), prefill(r, 16000)
    agent = ((r.get("agent") or {}).get("summary") or {}).get("write_tps")
    notes = []
    if r.get("invalid"):
        notes.append("invalid")
    if any(str(f).startswith("CONTENTION") for f in r.get("flags") or []):
        notes.append("contention")
    folder = os.path.relpath(os.path.dirname(path), HERE)
    link = f"[{os.path.basename(path)[:-5]}]({folder}/{os.path.basename(path)})"
    return (r["when"][:10], who or DASH, mac(r), model(r), quant(r), engine(r), version(r),
            num(dec, 1), num(gre, 1), num(p4.get("prefill_tps")), num(p16.get("prefill_tps")), num(p4.get("ttft_s"), 2),
            num(agent, 1), ", ".join(notes) or DASH, link, r["when"])


HEAD = ("Date", "Submitted by", "Mac", "Model", "Quant", "Engine", "Version", "Decode tok/s", "Greedy tok/s",
        "Prefill 4K tok/s", "Prefill 16K tok/s", "TTFT 4K s", "Agent tok/s", "Flags", "Record")


def first_authors():
    """{repo-relative path: author name of the commit that first added it}, for every file under bench/results."""
    if subprocess.run(["git", "-C", REPO, "rev-parse", "--is-shallow-repository"], capture_output=True, text=True).stdout.strip() != "false":
        sys.exit("bench/gen_readme.py needs the full git history (git fetch --unshallow, or checkout with fetch-depth: 0)")
    out = subprocess.run(["git", "-C", REPO, "log", "--diff-filter=A", "--no-renames", "--name-only", "--format=\x01%an", "--", "bench/results"],
                         capture_output=True, text=True, check=True).stdout
    who, cur = {}, None
    for line in out.splitlines():  # newest commit first, so the last one seen for a path is the first to add it
        if line.startswith("\x01"):
            cur = line[1:]
        elif line:
            who[line] = cur
    return who


def records():
    authors = first_authors()
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "**", "*.json"), recursive=True)):
        try:
            r = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not (isinstance(r, dict) and r.get("schema") and "engine" in r and "label" in r and "when" in r):
            continue  # not a run record
        try:
            rows.append(row(p, r, authors.get(os.path.relpath(p, REPO))))
        except (KeyError, TypeError, ValueError, IndexError) as e:
            sys.exit(f"{os.path.relpath(p, REPO)}: not a complete run record ({e!r})")
    rows.sort(key=lambda x: (x[15], x[2], x[3], x[5]), reverse=True)
    return rows


def render():
    rows = records()
    macs = sorted({x[2] for x in rows})
    engines = sorted({x[5].split(" ")[0] for x in rows})
    out = [BEGIN, "",
           f"{len(rows)} runs on {len(macs)} Macs ({'; '.join(macs)}), {rows[0][0] if rows else DASH} down to {rows[-1][0] if rows else DASH}, newest first.",
           "", "| " + " | ".join(HEAD) + " |", "| " + " | ".join(["---"] * 7 + ["---:"] * 6 + ["---", "---"]) + " |"]
    for x in rows:
        out.append("| " + " | ".join(cell(c) for c in x[:15]) + " |")
    out += ["", END]
    return "\n".join(out)


def unsubmitted(text):
    """The file with every table row's "Submitted by" cell blanked. The check ignores that column: a squash merge can
    give the file's first commit another author than the pull request's commits had, and the check would fail on main."""
    return re.sub(r"^(\| [^|]*? \| )[^|]*?( \| )", r"\1\2", text, flags=re.M)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if bench/README.md is stale (ignoring Submitted by)")
    a = ap.parse_args()
    text = open(README, encoding="utf-8").read()
    i, j = text.find(BEGIN), text.find(END)
    if i < 0 or j < i:
        sys.exit(f"{README}: the BEGIN/END markers are missing")
    new = text[:i] + render() + text[j + len(END):]
    if a.check:
        if unsubmitted(new) != unsubmitted(text):
            print("bench/README.md is stale: run `python3 bench/gen_readme.py` and commit the result", file=sys.stderr)
            return 1
        return 0
    open(README, "w", encoding="utf-8").write(new)
    print(f"wrote {os.path.relpath(README, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
