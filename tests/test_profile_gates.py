#!/usr/bin/env python3
"""Fail when a file outside the profile module gains an inline chip, core or RAM check, or a new predicate call.

The call-form baseline is tests/profile_gates_baseline.txt (path, count). The predicate baseline is
tests/profile_predicate_baseline.txt, the same shape, counting matrix_units() and the other named checks.
A count that goes down is a migration. A count that goes up, or a new file, fails. A copied local
(`int cores = g.core_count(); if (cores >= 40)`) is not seen; the call form is. A comment after // does not count.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests" / "profile_gates_baseline.txt"
PRED_BASELINE = ROOT / "tests" / "profile_predicate_baseline.txt"
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


def read_baseline(path: pathlib.Path = BASELINE) -> dict[str, int]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rel, n = line.split("\t")
        out[rel] = int(n)
    return out


# The named checks in device_profile.h. A new call outside the profile module fails the predicate baseline.
PREDICATES = (
    "matrix_units",
    "m1_max_32",
    "legacy_gpu",
    "wide_cores",
    "pre_m3_gpu",
    "simd_gpu",
    "ultra_gpu",
    "unconstrained_store",
)
PRED_RE = re.compile(r"\b(" + "|".join(PREDICATES) + r")\s*\(")


def predicate_hits(text: str) -> int:
    n = 0
    for line in text.splitlines():
        n += len(PRED_RE.findall(line.split("//", 1)[0]))
    return n


def scan_texts():
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
            yield rel, path.read_text(encoding="utf-8", errors="replace")


def predicate_file_counts() -> dict[str, int]:
    out = {}
    for rel, text in scan_texts():
        n = predicate_hits(text)
        if n:
            out[rel] = n
    return out


def predicate_totals() -> dict[str, int]:
    totals = {name: 0 for name in PREDICATES}
    for _, text in scan_texts():
        for line in text.splitlines():
            for name in PRED_RE.findall(line.split("//", 1)[0]):
                totals[name] += 1
    return totals


def baseline_rises(got: dict[str, int], want: dict[str, int], what: str) -> list[str]:
    bad = []
    for path, n in sorted(got.items()):
        old = want.get(path, 0)
        if n > old:
            bad.append(f"{path}: {old} -> {n} (new {what}; put it in device_profile.cpp)")
    for path in sorted(set(want) - set(got)):
        bad.append(f"{path}: baseline has {want[path]}, tree has 0 (update the baseline; a drop is fine)")
    return bad


def self_test() -> int:
    """The reviewer's case: a new file calling wide_cores, simd_gpu and matrix_units must fail."""
    probe = "bool x() { return wide_cores() || simd_gpu() || matrix_units(); }\n// wide_cores()\n"
    n = predicate_hits(probe)
    if n != 3:
        print(f"self-test: probe counted {n}, want 3")
        return 1
    if predicate_hits("// matrix_units()\n") != 0:
        print("self-test: a comment counted")
        return 1
    bad = baseline_rises({"src/model/zz_probe_gate.cpp": 3}, {}, "predicate call")
    if not any("zz_probe_gate.cpp" in line for line in bad):
        print("self-test: new file did not fail")
        return 1
    guard = predicate_hits("m1_max_32() || matrix_units(); // m1_max_32()")
    if guard != 2:
        print(f"self-test: guard and class predicates counted {guard}, want 2")
        return 1
    if not baseline_rises({"src/engine/engine.cpp": guard}, {"src/engine/engine.cpp": 1}, "predicate call"):
        print("self-test: a class call added beside an existing guard did not fail")
        return 1
    print("profile predicate self-test: ok (zz_probe_gate.cpp with wide_cores/simd_gpu/matrix_units fails)")
    return 0


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

# A name predicate the call sites still read. Each id must be a kGuard row, not a kName row:
# none of them cites a 10% win the wider class misses.
GUARD_IDS = (
    "m1_max_32",
    "ultra_gpu",
    "pre_m3_gpu",
    "unconstrained_store",
)


def table_ids() -> set[str]:
    text = (ROOT / "src/engine/device_profile.cpp").read_text(encoding="utf-8")
    start = text.find("const DefaultRow kDefaults[]")
    end = text.find("};", start)
    return set(re.findall(r'\{"([a-z0-9_]+)"', text[start:end]))


def site_label(n: int) -> str:
    return f"{n} site;" if n == 1 else f"{n} sites;"


def table_count_bad(totals: dict[str, int]) -> list[str]:
    text = (ROOT / "src/engine/device_profile.cpp").read_text(encoding="utf-8")
    start = text.find("const DefaultRow kDefaults[]")
    table = text[start : text.find("};", start)]
    bad = []
    for name, n in totals.items():
        row = re.search(rf'\{{"{name}".*?(?=\{{"|$)', table, re.S)
        body = row.group(0) if row else ""
        if not body:
            bad.append(f"{name}: no table row")
        elif not re.search(rf"(?<!\d){re.escape(site_label(n))}", body):
            bad.append(f"{name}: table does not say {site_label(n)} (scan has {n})")
    return bad


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        return self_test()
    if self_test() != 0:
        return 1
    got = counts()
    pred = predicate_file_counts()
    if len(sys.argv) > 1 and sys.argv[1] == "--write-baseline":
        lines = [
            "# path, call-form hits. A rise fails tests/test_profile_gates.py. A drop to zero fails until this row is removed.",
            *[f"{p}\t{got[p]}" for p in sorted(got)],
        ]
        BASELINE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {len(lines)} rows to {BASELINE}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--write-predicate-baseline":
        lines = [
            "# path, predicate-call hits outside the profile module. A rise fails tests/test_profile_gates.py. A drop to zero fails until this row is removed.",
            *[f"{p}\t{pred[p]}" for p in sorted(pred)],
        ]
        PRED_BASELINE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {len(lines) - 1} rows to {PRED_BASELINE}")
        return 0
    want = read_baseline()
    bad = baseline_rises(got, want, "inline chip/core/RAM check")
    # A drop in a file that still has some hits is fine and does not fail. A file that dropped to zero
    # is reported above so the baseline does not keep a stale allow.
    ids = table_ids()
    missing = [i for i in CLASS_GATE_IDS if i not in ids]
    if missing:
        bad.append("class gate has no default-table row: " + ", ".join(missing))
    text = (ROOT / "src/engine/device_profile.cpp").read_text(encoding="utf-8")
    start = text.find("const DefaultRow kDefaults[]")
    table = text[start : text.find("};", start)]
    unguarded = [i for i in GUARD_IDS if not re.search(rf'\{{"{i}".*DefaultScope::kGuard', table)]
    if unguarded:
        bad.append("name predicate has no kGuard row: " + ", ".join(unguarded))
    totals = predicate_totals()
    bad.extend(baseline_rises(pred, read_baseline(PRED_BASELINE), "predicate call"))
    bad.extend(table_count_bad(totals))
    if bad:
        print("profile gate lint failed:")
        print("\n".join(bad))
        return 1
    print(
        f"profile gate lint: ok ({sum(got.values())} call-form hits, {len(got)} files, "
        f"{sum(totals.values())} predicate calls, {len(CLASS_GATE_IDS)} class gates, {len(GUARD_IDS)} guard rows)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
