#!/usr/bin/env python3
"""PR size check: counts the lines a reviewer has to read (added + deleted) and compares them with a cap.

One change per PR (CONTRIBUTING.md) is the rule; this makes a PR that is several changes visible. Data that is
reviewed by what produced it, not line by line, is listed separately and not counted: third-party sources, recorded
benchmark results, frozen prompts and token ids, images. Code, tests and docs prose count.

Environment: BASE_SHA, HEAD_SHA (commits in this checkout), LIMIT (default 1000). The override is a signal to the
reviewer, not a way for the author to skip the check: HAS_OVERRIDE_LABEL ("true" when the PR carries `size/override`,
which only people with triage rights can add) together with a `Size override: <why>` line in PR_BODY.
GITHUB_STEP_SUMMARY is written when set. Exit 1 when over the cap without an override, 2 on an error.

The script reads the diff as data (git diff --numstat) and runs from the base branch (the workflow is
pull_request_target), so a PR cannot edit its own check.
"""
import fnmatch
import os
import subprocess
import sys

# first match wins; fnmatch's * crosses /, so "dir/*" is the whole subtree
UNCOUNTED = [
    ("third-party sources", ["third_party/*"]),
    ("recorded results", ["bench/results/*", "bench/community/results/*"]),
    ("frozen prompts, ids and corpora", ["bench/prompts/*", "bench/quality/ids_*", "bench/quality/src/*",
                                         "bench/shortpf/p*.txt", "bench/harness/corpus/*", "bench/harness/agent-traces/*"]),
    ("images", ["docs/assets/*", "*.png", "*.jpg", "*.svg", "*.gif"]),
]


def category(path):
    for name, patterns in UNCOUNTED:
        if any(fnmatch.fnmatchcase(path, p) for p in patterns):
            return name
    return None


def numstat(base, head):
    # three dots: the PR's own changes since it branched, not what main gained meanwhile
    out = subprocess.run(["git", "diff", "--numstat", "--no-renames", f"{base}...{head}"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"git diff {base}...{head} failed (a shallow checkout? fetch-depth must be 0):\n{out.stderr}")
    for line in out.stdout.splitlines():
        added, deleted, path = line.split("\t", 2)
        yield path, (0 if added == "-" else int(added)) + (0 if deleted == "-" else int(deleted))  # binary: 0 lines


def main():
    base, head = os.environ.get("BASE_SHA"), os.environ.get("HEAD_SHA")
    if not base or not head:
        sys.exit("BASE_SHA and HEAD_SHA are required")
    limit = int(os.environ.get("LIMIT") or 1000)
    why = [l for l in (os.environ.get("PR_BODY") or "").splitlines() if l.strip().lower().startswith("size override:")]
    override = os.environ.get("HAS_OVERRIDE_LABEL") == "true" and any(l.split(":", 1)[1].strip() for l in why)
    counted, skipped, biggest = 0, {}, []
    for path, n in numstat(base, head):
        cat = category(path)
        if cat:
            skipped[cat] = skipped.get(cat, 0) + n
        else:
            counted += n
            biggest.append((n, path))
    over = counted > limit
    lines = [f"## PR size: {counted} counted lines (cap {limit})", ""]
    if skipped:
        lines += ["Not counted: " + ", ".join(f"{c} ({n})" for c, n in sorted(skipped.items())), ""]
    if over:
        lines += ["Largest counted files:", ""] + [f"- `{p}`: {n}" for n, p in sorted(biggest, reverse=True)[:8]] + [""]
        if override:
            lines.append("Over the cap, but allowed: `size/override` label and a `Size override:` line.")
        else:
            lines.append("Over the cap: split it into PRs of one change each. If it can't be split, a *reviewer* "
                         "adds the `size/override` label and the PR description gets a `Size override: <why>` line.")
    text = "\n".join(lines)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")
    return 1 if over and not override else 0


if __name__ == "__main__":
    sys.exit(main())
