#!/usr/bin/env python3
"""docs/models.md against src/hub/catalog.cpp (tools/models_table.py): the committed table matches the catalog, and the
check fails when a revision, an alias, a drafter or a download size changes without the table. CPU only, stdlib only.

usage: test_models_table.py
"""
import os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
TOOL = os.path.join(ROOT, "tools", "models_table.py")
CATALOG = os.path.join(ROOT, "src", "hub", "catalog.cpp")
TABLE = os.path.join(ROOT, "docs", "models.md")
FAILED = []


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  ({detail})"))
    if not ok:
        FAILED.append(name)


def run(*args):
    return subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)


def main():
    cat = open(CATALOG).read()
    r = run("--check")
    check(r.returncode == 0, "committed table matches the catalog", r.stdout[-400:] + r.stderr[-200:])
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "models.md")
        open(out, "w").write(open(TABLE).read())
        # each edit is a drift a catalog PR could make (the 27B drafter's revision is kapitsa#1107's case); they
        # rewrite the first match of a pattern, so the test does not pin today's values
        def drafter_rev(t):
            i = t.index("Companion dflash27()")
            return re.sub(r'(\.revision = ")[0-9a-f]{40}', r"\g<1>" + "1" * 40, t[i:], count=1), i

        def edit(t, pat, repl):
            out, n = re.subn(pat, repl, t, count=1)
            assert n == 1, pat
            return out

        def drafter(t):
            tail, i = drafter_rev(t)
            return t[:i] + tail

        edits = {
            "a drafter's revision": drafter,
            "an alias": lambda t: edit(t, r'\.aliases = \{', '.aliases = {"zz-new", '),
            "a download size": lambda t: edit(t, r"\.download_gb = (\d+)", r".download_gb = 9\1"),
            "an MTP head's revision": lambda t: edit(t, r'(?s)(Companion mtp27\(\).*?\.revision = ")[0-9a-f]{40}', r"\g<1>" + "2" * 40),
            "a weights repo": lambda t: edit(t, r'(\.repo = "lmstudio-community/[^"]+)', r"\1-v2"),
            "a KQuant tier slug": lambda t: edit(t, r'(\.repo = "nace-ai/[^"]+-MLX-Q)4(-KQuant")', r"\g<1>5\2"),
        }
        for what, fn in edits.items():
            alt = os.path.join(d, "catalog.cpp")
            changed = fn(cat)
            check(changed != cat, f"the test's edit changes the catalog ({what})")
            open(alt, "w").write(changed)
            r = run("--check", "--catalog", alt, "--out", out)
            check(r.returncode == 1, f"--check fails when {what} changes", f"rc={r.returncode}")
            run("--catalog", alt, "--out", out)
            r = run("--check", "--catalog", alt, "--out", out)
            check(r.returncode == 0, f"a rewrite brings the table back after {what} changes")
            open(out, "w").write(open(TABLE).read())
        # a catalog the parser cannot account for fails instead of writing a short table
        alt = os.path.join(d, "catalog.cpp")
        open(alt, "w").write(re.sub(r'(\.revision = ")[0-9a-f]{40}', r"\g<1>c43ee1d", cat, count=1))
        r = run("--check", "--catalog", alt, "--out", out)
        check(r.returncode != 0 and "pinned revision" in (r.stderr + r.stdout), "a variant without a full revision fails the run")
    print("FAILED: " + ", ".join(FAILED) if FAILED else "all passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
