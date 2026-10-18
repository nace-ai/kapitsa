#!/usr/bin/env python3
"""The Apple silicon Macs `kapitsa bench` knows by name: model identifier (`sysctl hw.model`, e.g. Mac16,7) -> Apple's
name for the model and the year it came out, read from Apple's own "Identify your Mac model" support pages. Stdlib only.

  python3 tools/mac_models.py              fetch Apple's pages and write bench/community/mac-models.json
  python3 tools/mac_models.py --check      fetch them and compare with bench/community/mac-models.json (exit 1 if
                                           they differ: a new Mac, or a name Apple changed)
  python3 tools/mac_models.py --html DIR   read saved pages (DIR/macbookpro.html, ...) instead of fetching them

The JSON is embedded in the kapitsa binary at build time (`kapitsa bench` names the Mac from it; an identifier it does
not list is shown as is) and tools/community_check.py checks submitted results against it. Each entry comes from one
section of a page: its heading (the name), its "Model Identifier" line, and its "Year introduced" line, else the year in
the name ("MacBook Pro (14-inch, Nov 2023)": 2023). Apple silicon identifiers are the Mac<N>,<M> ones (all Macs since
2022) and the six families of 2020-2021 listed in M1_IDS. An identifier on two sections (Mac Pro and Mac Pro (Rack),
both Mac14,8) takes the first.
"""
import argparse, datetime, html, json, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "bench", "community", "mac-models.json")
PAGES = [  # (file stem, Apple support article)
    ("macbookpro", "108052"),   # Identify your MacBook Pro model
    ("macbookair", "102869"),   # Identify your MacBook Air model
    ("macmini", "102852"),      # Identify your Mac mini model
    ("imac", "108054"),         # Identify your iMac model
    ("macstudio", "102231"),    # Identify your Mac Studio model
    ("macpro", "102887"),       # Identify your Mac Pro model
]
URL = "https://support.apple.com/en-us/{}"
# Apple silicon Macs from before the Mac<N>,<M> identifiers (M1, M1 Pro, M1 Max)
M1_IDS = {"MacBookAir10,1", "MacBookPro17,1", "MacBookPro18,1", "MacBookPro18,2", "MacBookPro18,3", "MacBookPro18,4",
          "Macmini9,1", "iMac21,1", "iMac21,2"}


def apple_silicon(identifier):
    return identifier in M1_IDS or re.fullmatch(r"Mac\d+,\d+", identifier) is not None


def fetch(article):
    # without Accept-Language the site answers with its locale chooser instead of the article
    req = urllib.request.Request(URL.format(article), headers={"Accept-Language": "en-US,en;q=0.9", "Accept": "text/html",
                                                               "User-Agent": "kapitsa-mac-models/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def text(fragment):
    s = re.sub(r"<[^>]+>", "", fragment)
    return re.sub(r"\s+", " ", html.unescape(s).replace("\xa0", " ")).strip()


def sections(page):
    """Each model section of a page in order: {name, ids, year_introduced}."""
    out, cur = [], None
    for m in re.finditer(r'<h2 class="gb-header">(.*?)</h2>|<p class="gb-paragraph">(.*?)</p>', page, re.S):
        if m.group(1) is not None:
            cur = {"name": text(m.group(1)), "ids": [], "year_introduced": None}
            out.append(cur)
        elif cur is not None:
            t = text(m.group(2))
            if t.startswith("Model Identifier"):
                cur["ids"] += [x.strip() for x in re.split(r"[,;]\s*(?=[A-Za-z])", t.split(":", 1)[1]) if x.strip()]
            elif t.startswith("Year introduced"):
                cur["year_introduced"] = t.split(":", 1)[1].strip()
    return [s for s in out if s["ids"]]


def models(pages):
    """[{identifier, name, year}] of every Apple silicon Mac on the pages, in page order (newest first)."""
    seen, out, problems = {}, [], []
    for stem, page in pages:
        for s in sections(page):
            ids = [i for i in s["ids"] if apple_silicon(i)]
            if not ids:
                continue
            named = re.findall(r"\b(20\d\d)\b", s["name"])
            year = s["year_introduced"] or (named[-1] if named else None)
            if not (year and re.fullmatch(r"20\d\d", year)):
                problems.append(f"{stem}: {s['name']} ({', '.join(ids)}) has no year")
                continue
            if named and named[-1] != year:
                problems.append(f"{stem}: {s['name']} says {named[-1]}, its 'Year introduced' {year}")
            for i in ids:
                if i in seen:
                    if seen[i] != s["name"]:
                        print(f"note: {i} is also {s['name']!r} (kept {seen[i]!r})", file=sys.stderr)
                    continue
                seen[i] = s["name"]
                out.append({"identifier": i, "name": s["name"], "year": int(year)})
    if problems:
        raise SystemExit("can't use these sections:\n  " + "\n  ".join(problems))
    return out


def dump(entries):
    """The JSON file: one model per line, so a new Mac is a one-line diff."""
    head = {"about": "Apple silicon Macs: model identifier (sysctl hw.model) -> Apple's name for it and the year it came "
                     "out, from Apple's 'Identify your Mac model' pages. Written by tools/mac_models.py; do not edit by hand.",
            "sources": {stem: URL.format(article) for stem, article in PAGES},
            "checked": datetime.date.today().isoformat()}
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    body = json.dumps(head, indent=1, ensure_ascii=False)[:-2]
    return body + ',\n "models": [\n  ' + ",\n  ".join(lines) + "\n ]\n}\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="compare Apple's pages with the JSON file; write nothing")
    ap.add_argument("--html", metavar="DIR", help="read saved pages DIR/<stem>.html instead of fetching them")
    ap.add_argument("--out", default=OUT, help="the JSON file (default: bench/community/mac-models.json)")
    a = ap.parse_args()
    pages = []
    for stem, article in PAGES:
        if a.html:
            with open(os.path.join(a.html, stem + ".html"), encoding="utf-8") as fh:
                pages.append((stem, fh.read()))
        else:
            pages.append((stem, fetch(article)))
    entries = models(pages)
    if a.check:
        with open(a.out, encoding="utf-8") as fh:
            have = {m["identifier"]: m for m in json.load(fh)["models"]}
        want = {m["identifier"]: m for m in entries}
        diffs = [f"new on Apple's pages: {want[i]}" for i in want if i not in have]
        diffs += [f"no longer on Apple's pages: {have[i]}" for i in have if i not in want]
        diffs += [f"changed: {have[i]} -> {want[i]}" for i in want if i in have and have[i] != want[i]]
        for d in diffs:
            print(d)
        print(f"{len(want)} Apple silicon identifiers on Apple's pages, {len(have)} in {os.path.relpath(a.out)}: "
              + ("the same" if not diffs else f"{len(diffs)} differences"))
        return 1 if diffs else 0
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(dump(entries))
    print(f"{len(entries)} Apple silicon identifiers -> {os.path.relpath(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
