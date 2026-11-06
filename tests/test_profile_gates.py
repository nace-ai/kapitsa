#!/usr/bin/env python3
"""Fail when a file outside the profile module gains an inline chip, core or RAM check.

The baseline is tests/profile_gates_baseline.txt (path, count). A count that goes down is a
migration. A count that goes up, or a new file, is a gate that belongs in src/engine/device_profile.cpp.
A copied local (`int cores = g.core_count(); if (cores >= 40)`) is not seen; the call form is.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests" / "profile_gates_baseline.txt"
SCAN = ("src", "tools", "tests")
EXEMPT = {
    "src/engine/device_profile.cpp",
    "src/engine/device_profile.h",
    "src/engine/device_profile_live.cpp",
    "tests/device_profile_test.cpp",
    "tests/test_profile_gates.py",
}
# The call form the model files use today. A comment after // does not count.
PATTERNS = [
    re.compile(r"apple_family\s*\("),
    re.compile(r"apple10\s*\("),
    re.compile(r"m4_base\s*\("),
    re.compile(r'name\(\)\s*==\s*"Apple'),
    re.compile(r"name\(\)\.find\("),
    re.compile(r"core_count\(\)\s*(==|!=|>=|<=|>|<)"),
    re.compile(r"ram_bytes\(\)\s*(==|!=|>=|<=|>|<)"),
    re.compile(r"24\s*<<\s*30"),
    re.compile(r"48\s*<<\s*30"),
]


def hits(text: str) -> int:
    n = 0
    for line in text.splitlines():
        code = line.split("//", 1)[0]
        n += sum(1 for p in PATTERNS if p.search(code))
    return n


def counts() -> dict[str, int]:
    out = {}
    for top in SCAN:
        base = ROOT / top
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix not in {".cpp", ".h", ".mm", ".metal"}:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel in EXEMPT or "/build/" in rel:
                continue
            n = hits(path.read_text(encoding="utf-8", errors="replace"))
            if n:
                out[rel] = n
    return out


def read_baseline() -> dict[str, int]:
    out = {}
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path, n = line.split("\t")
        out[path] = int(n)
    return out


# A class-scoped NI_* default must have a row in src/engine/device_profile.cpp. Add the id here when you add the row.
CLASS_GATE_IDS = (
    "pf_wave",
    "kflash_hist",
    "gemm_slices",
    "hc_wide",
    "moe4_dx",
    "tree_g6",
    "kflash_narrow",
)


def table_ids() -> set[str]:
    text = (ROOT / "src/engine/device_profile.cpp").read_text(encoding="utf-8")
    start = text.find("const DefaultRow kDefaults[]")
    end = text.find("};", start)
    return set(re.findall(r'\{"([a-z0-9_]+)"', text[start:end]))


def main() -> int:
    got = counts()
    if len(sys.argv) > 1 and sys.argv[1] == "--write-baseline":
        lines = [
            "# path, call-form hits. A rise fails tests/test_profile_gates.py. A drop to zero fails until this row is removed.",
            *[f"{p}\t{got[p]}" for p in sorted(got)],
        ]
        BASELINE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {len(lines)} rows to {BASELINE}")
        return 0
    want = read_baseline()
    bad = []
    for path, n in sorted(got.items()):
        old = want.get(path, 0)
        if n > old:
            bad.append(f"{path}: {old} -> {n} (new inline chip/core/RAM check; put it in device_profile.cpp)")
    for path in sorted(set(want) - set(got)):
        bad.append(f"{path}: baseline has {want[path]}, tree has 0 (update the baseline; a drop is fine)")
    # A drop in a file that still has some hits is fine and does not fail. A file that dropped to zero
    # is reported above so the baseline does not keep a stale allow.
    missing = [i for i in CLASS_GATE_IDS if i not in table_ids()]
    if missing:
        bad.append("class gate has no default-table row: " + ", ".join(missing))
    if bad:
        print("profile gate lint failed:")
        print("\n".join(bad))
        return 1
    print(
        f"profile gate lint: ok ({sum(got.values())} call-form hits, {len(got)} files, "
        f"{len(CLASS_GATE_IDS)} class gates)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
