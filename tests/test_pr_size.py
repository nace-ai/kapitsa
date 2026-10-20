#!/usr/bin/env python3
"""The PR size check (.github/scripts/pr_size.py) on a throwaway git repository: what counts, what does not, the cap,
and that the override needs both the label and a reason. CPU only, stdlib only.

usage: test_pr_size.py
"""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", ".github", "scripts", "pr_size.py")
FAILED = []


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  ({detail})"))
    if not ok:
        FAILED.append(name)


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def write(repo, path, lines):
    full = os.path.join(repo, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write("".join(f"line {i}\n" for i in range(lines)))


def run(repo, base, head, **env):
    e = {k: v for k, v in os.environ.items() if k not in ("GITHUB_STEP_SUMMARY",)}
    e.update(BASE_SHA=base, HEAD_SHA=head, **env)
    p = subprocess.run([sys.executable, SCRIPT], cwd=repo, env=e, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


with tempfile.TemporaryDirectory() as repo:
    git(repo, "init", "-q", "-b", "main")
    write(repo, "src/a.cpp", 10)
    git(repo, "add", "-A"); git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD")

    git(repo, "checkout", "-qb", "pr")
    write(repo, "src/a.cpp", 30)                  # the first 10 lines stay: 20 added
    write(repo, "tests/test_x.py", 50)            # tests count: a change without them is not smaller
    write(repo, "third_party/lib/big.h", 5000)    # not counted
    write(repo, "bench/results/run/out.json", 700)  # not counted
    write(repo, "docs/assets/pic.svg", 300)       # not counted
    git(repo, "add", "-A"); git(repo, "commit", "-qm", "change")
    head = git(repo, "rev-parse", "HEAD")

    rc, out = run(repo, base, head)
    check(rc == 0 and "70 counted lines" in out, "counts code and tests, skips third-party, results and images", out)
    check("third-party sources (5000)" in out and "recorded results (700)" in out, "lists what it skipped", out)

    rc, out = run(repo, base, head, LIMIT="50")
    check(rc == 1 and "src/a.cpp" in out and "Over the cap" in out, "over the cap fails and names the files", out)

    rc, _ = run(repo, base, head, LIMIT="50", HAS_OVERRIDE_LABEL="true")
    check(rc == 1, "the label alone is not an override")
    rc, _ = run(repo, base, head, LIMIT="50", PR_BODY="Size override:   \n")
    check(rc == 1, "a reason alone is not an override")
    rc, _ = run(repo, base, head, LIMIT="50", HAS_OVERRIDE_LABEL="true", PR_BODY="x\nSize override: one rename, mechanical\n")
    check(rc == 0, "the label and a reason together are")

    # the base moving on must not count: three dots compare with the merge base
    git(repo, "checkout", "-q", "main"); write(repo, "src/other.cpp", 400)
    git(repo, "add", "-A"); git(repo, "commit", "-qm", "main moves on")
    rc, out = run(repo, git(repo, "rev-parse", "main"), head)
    check(rc == 0 and "70 counted lines" in out, "only the PR's own changes count", out)

    rc, out = run(repo, "0" * 40, head)
    check(rc != 0, "a bad base is an error, not a pass", out)

print("\nALL PASSED" if not FAILED else f"\nFAIL: {len(FAILED)} failed: {FAILED}")
sys.exit(1 if FAILED else 0)
