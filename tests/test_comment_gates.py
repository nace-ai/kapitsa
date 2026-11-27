#!/usr/bin/env python3
"""Fail when a comment or doc line anywhere in the repo gives "it was only measured here" (or "nothing was run, so") as the reason for a narrow default.

AGENTS.md: defaults follow the concept, not the measured cell. These phrases are how the old rule came back in rounds
1 to 3 of #368: "not measured, so ...", "only after ...", "measured on ... only", "until ... A/B", and "nothing has been run on it, so ...". A comment that says
what the code does today ("on today for the 35B shape only") is fine; one that gives the measurement as the reason is not.

Scope: every tracked text file except third_party/ and bench/results/ (data). Code files (.cpp .h .mm .metal) are read as
comments only; every other text file is read line by line.

tests/comment_gates_allowed.txt lists the lines that match and are a different thing (an order of events, a fact about
a run, or a gate that cites a break). Narrow gates with no run behind them are listed in their own section, each
waiting on #369. A match that is not listed fails. Add a line there only when it is not a reason to limit a default, and
say why in the PR.
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
    re.compile(r"\b(?:not|never|nothing has been|nothing was)\s+(?:been\s+)?(?:run|tested|measured|tried)\b[^.;|]*?,\s*so\b", re.I),
]
COMMENT_ONLY_SUFFIXES = {".cpp", ".h", ".mm", ".metal"}
SKIP_PREFIXES = ("third_party/", "bench/results/")
SKIP_FILES = {"tests/test_comment_gates.py", "tests/comment_gates_allowed.txt"}


def comment_part(path: str, line: str) -> str:
    """A C++ or Metal file's comment text; any other file's line as it is."""
    if not path.endswith(tuple(COMMENT_ONLY_SUFFIXES)):
        return line
    s = line.strip()
    if s.startswith(("*", "/*")):
        return s
    return line.split("//", 1)[1] if "//" in line else ""


def matches(text: str) -> bool:
    return any(p.search(text) for p in PATTERNS)


def tracked() -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split("\n")
    return [f for f in out if f and f not in SKIP_FILES and not f.startswith(SKIP_PREFIXES)]


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


def stale_entries(found: list[tuple[str, int, str]], listed: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """Allowlist lines that no longer match any line: the list must not rot."""
    seen = {(f, t) for f, _, t in found}
    return sorted(listed - seen)


def find() -> list[tuple[str, int, str]]:
    found = []
    for f in tracked():
        try:
            lines = (ROOT / f).read_text(encoding="utf-8").split("\n")
        except (OSError, UnicodeDecodeError):  # an image or other binary
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
        "// same shape and drafter and nothing has been run on it, so the gate still asks for not Clover",
        "Ollama was not run on that pack, so its cell is a dash.",
        "// has never been tested there, so it stays off",
    ]
    fine = [
        "// on today for the 35B shape only",
        "// the numbers are from the M5 Max; the default covers the M5 class",
        "int only_after = 1;  // the count",
        "// not measured here, but the same kernel",
        "code only after the guard;",
        "// the fused path was run on the M5 Pro, so the class takes it",
        "// nothing was run on other models or on English chat.",
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
    if stale_entries([("a.cpp", 1, "// x")], {("a.cpp", "// x"), ("a.cpp", "// gone")}) != [("a.cpp", "// gone")]:
        print("self-test: a listed line that matches nothing must be reported")
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
    found = find()
    new = [(f, i, t) for f, i, t in found if not is_allowed(f, t, ok)]
    stale = stale_entries(found, ok)
    for f, i, t in new:
        print(f"{f}:{i}: {t[:150]}")
    for f, t in stale:
        print(f"{f}: allowlisted line no longer matches (remove it from tests/comment_gates_allowed.txt): {t[:110]}")
    if stale and not new:
        return 1
    if new:
        print(
            f"\n{len(new)} line(s) give 'only measured here' as the reason for a narrow default. Say what the code does today, "
            "or the class the default covers (AGENTS.md: defaults follow the concept). tests/comment_gates_allowed.txt is only "
            "for a match that is not a gate."
        )
        return 1
    print(f"test_comment_gates: ok ({len(found)} listed matches that are not gates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
