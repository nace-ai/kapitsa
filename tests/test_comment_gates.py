#!/usr/bin/env python3
"""Fail when a comment in src/ or a line in docs/ gives "it was only measured here" as the reason for a narrow default.

AGENTS.md: defaults follow the concept, not the measured cell. These phrases are how the old rule came back in rounds
1 to 3 of #368: "not measured, so ...", "only after ...", "measured on ... only", "until ... A/B". A comment that says
what the code does today ("on today for the 35B shape only") is fine; one that gives the measurement as the reason is not.

tests/comment_gates_allowed.txt lists the lines that match and are a different thing (an order of events, not a gate).
A match that is not listed fails. Add a line there only when it is not a reason to limit a default, and say why in the PR.
Run `python3 tests/test_comment_gates.py --self-test` to check the patterns themselves.
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOWED = ROOT / "tests" / "comment_gates_allowed.txt"
PATTERNS = [
    re.compile(r"not measured,? so", re.I),
    re.compile(r"\bonly after\b", re.I),
    re.compile(r"\bmeasured on\b[^.;|]*?\bonly\b", re.I),
    re.compile(r"\buntil\b[^.;|]*?\bA/B\b", re.I),
]
CODE_SUFFIXES = {".cpp", ".h", ".mm", ".metal", ".py"}
SKIP_PREFIXES = ("docs/research/",)
SKIP_FILES = {"docs/numerical-history.md", "tests/test_comment_gates.py"}


def comment_part(path: str, line: str) -> str:
    """A code file's comment text, or a doc line as it is."""
    if not path.endswith(tuple(CODE_SUFFIXES)):
        return line
    if path.endswith(".py"):
        return line.split("#", 1)[1] if "#" in line else ""
    s = line.strip()
    if s.startswith(("*", "/*")):
        return s
    return line.split("//", 1)[1] if "//" in line else ""


def matches(text: str) -> bool:
    return any(p.search(text) for p in PATTERNS)


def tracked() -> list[str]:
    out = subprocess.check_output(["git", "ls-files", "src", "docs"], cwd=ROOT, text=True).split("\n")
    keep = []
    for f in out:
        if not f or f in SKIP_FILES or f.startswith(SKIP_PREFIXES):
            continue
        if pathlib.PurePosixPath(f).suffix in CODE_SUFFIXES | {".md"}:
            keep.append(f)
    return keep


def allowed() -> set[tuple[str, str]]:
    """(path, stripped line) pairs: a copy of an allowed sentence in another file is not allowed."""
    out = set()
    for line in ALLOWED.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            path, _, text = line.partition("\t")
            out.add((path.strip(), text.strip()))
    return out


def is_allowed(path: str, text: str, listed: set[tuple[str, str]]) -> bool:
    return (path, text) in listed


def find() -> list[tuple[str, int, str]]:
    found = []
    for f in tracked():
        try:
            lines = (ROOT / f).read_text(encoding="utf-8", errors="replace").split("\n")
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            if matches(comment_part(f, line)):
                found.append((f, i, line.strip()))
    return found


def self_test() -> int:
    bad = [
        "// A model joins this list only after its own interleaved A/B",
        "// same 35B-A3B shape; not measured, so not kQwen35A3b",
        "Measured on the M5 Max only: other Macs keep it off.",
        "// keeps one tile until a Mac's A/B says a wider count pays",
        " * only after the A/B",
    ]
    fine = [
        "// on today for the 35B shape only",
        "// the numbers are from the M5 Max; the default covers the M5 class",
        "int only_after = 1;  // the count",
        "// not measured here, but the same kernel",
        "code only after the guard;",
    ]
    ok = True
    for t in bad:
        if not matches(comment_part("x.cpp", t)) and not matches(comment_part("x.md", t)):
            print("self-test: not caught:", t)
            ok = False
    for t in fine:
        if matches(comment_part("x.cpp", t)):
            print("self-test: wrongly caught:", t)
            ok = False
    # the allowlist is keyed on the path: the same line in another file is not allowed
    listed = {("src/a.cpp", "// only after X")}
    if not is_allowed("src/a.cpp", "// only after X", listed) or is_allowed("src/b.cpp", "// only after X", listed):
        print("self-test: the allowlist must match its own path and no other")
        ok = False
    print("self-test:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        for f, i, t in find():
            print(f"{f}:{i}\t{t}")
        return 0
    ok = allowed()
    new = [(f, i, t) for f, i, t in find() if not is_allowed(f, t, ok)]
    for f, i, t in new:
        print(f"{f}:{i}: {t[:150]}")
    if new:
        print(
            f"\n{len(new)} line(s) give 'only measured here' as the reason for a narrow default. Say what the code does today, "
            "or the class the default covers (AGENTS.md: defaults follow the concept). tests/comment_gates_allowed.txt is only "
            "for a match that is not a gate."
        )
        return 1
    print(f"test_comment_gates: ok ({len(find())} listed matches that are not gates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
