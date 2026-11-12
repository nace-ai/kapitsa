#!/usr/bin/env python3
"""Check that every nace-ai KQuant Hugging Face id this checkout names resolves directly, and that the catalog's
pinned revisions still hold the same files.

1. Every nace-ai KQuant repo id in the tracked docs, sources, tests and tools (not bench/: those files are frozen
   records of what ran) must answer HTTP 200 on the model API with no redirect.
2. Every `.repo`/`.revision` pin of a KQuant pack in src/hub/catalog.cpp must resolve (tree of that revision, 200, no
   redirect). With --base-ref, the same pins at that ref are fetched too (following redirects, as the old ids need)
   and each pack's safetensors must have the same paths, sizes and LFS sha256 in both lists.

Anonymous requests only, so what passes is what a user without a token gets. Stdlib only.
    python3 tools/check_kquant_ids.py --base-ref origin/main
"""
import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request

HUB = "https://huggingface.co"
ID_RE = re.compile(r"nace-ai/[A-Za-z0-9._-]*KQuant[A-Za-z0-9._-]*")
PIN_RE = re.compile(r'\.repo = "(nace-ai/[A-Za-z0-9._-]*KQuant[A-Za-z0-9._-]*)",\s*\.revision = "([0-9a-f]{40})"')
CATALOG = "src/hub/catalog.cpp"
MARK = "redirected old id"
SCAN = ["README.md", "docs", "src", "tests", "tools"]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def get(url, follow):
    opener = urllib.request.build_opener(*([] if follow else [NoRedirect()]))
    try:
        with opener.open(urllib.request.Request(url, headers={"User-Agent": "kapitsa-check-ids"}), timeout=60) as r:
            return r.status, "", json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Location", ""), None


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def ids_in_tree():
    ids = set()
    for line in git("grep", "-h", "-E", ID_RE.pattern, "--", *SCAN).splitlines():
        if MARK not in line:  # the old ids Kapitsa still accepts as input are marked, and redirect on purpose
            ids.update(ID_RE.findall(line))
    return sorted(ids)


def pins(ref):
    text = git("show", f"{ref}:{CATALOG}") if ref else open(CATALOG).read()
    return PIN_RE.findall(text)


def safetensors(repo, rev, follow):
    code, loc, tree = get(f"{HUB}/api/models/{repo}/tree/{rev}?recursive=1", follow)
    if code != 200:
        return code, loc, None
    files = {e["path"]: (e.get("size"), (e.get("lfs") or {}).get("oid")) for e in tree if e["path"].endswith(".safetensors")}
    return code, loc, files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ref", help="a git ref whose catalog pins must hold the same files (e.g. origin/main)")
    args = ap.parse_args()
    bad = 0

    print("== ids: GET /api/models/<id>, redirects not followed")
    for i in ids_in_tree():
        code, loc, _ = get(f"{HUB}/api/models/{i}", False)
        ok = code == 200
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {code} {i}" + (f" -> {loc}" if loc else ""))

    print("== pins: tree of each catalog (repo, revision), redirects not followed")
    now = pins(None)
    after = []
    for repo, rev in now:
        code, loc, files = safetensors(repo, rev, False)
        ok = code == 200 and bool(files)
        bad += not ok
        after.append(files)
        first = sorted(files)[0] if files else "-"
        print(f"{'ok  ' if ok else 'FAIL'} {code} {repo}@{rev[:8]} {len(files or {})} safetensors; first {first} "
              f"sha256 {((files or {}).get(first) or (0, ''))[1]}" + (f" -> {loc}" if loc else ""))

    if args.base_ref:
        print(f"== same files: pins at {args.base_ref} (redirects followed) against the pins above")
        before = pins(args.base_ref)
        if len(before) != len(now) or [r for _, r in before] != [r for _, r in now]:
            bad += 1
            print(f"FAIL the pin lists differ in length or revisions: {len(before)} at {args.base_ref}, {len(now)} here")
        else:
            for (b_repo, rev), (n_repo, _), n_files in zip(before, now, after):
                code, _, b_files = safetensors(b_repo, rev, True)
                ok = code == 200 and b_files == n_files and bool(n_files)
                bad += not ok
                print(f"{'ok  ' if ok else 'FAIL'} {b_repo}@{rev[:8]} -> {n_repo}: {len(n_files or {})} safetensors, "
                      f"{'same paths, sizes and sha256' if ok else 'DIFFERENT or missing'}")

    print("FAILED" if bad else "ALL OK", f"({bad} failing)" if bad else "")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
