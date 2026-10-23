#!/usr/bin/env python3
"""bench/gen_readme.py on records that miss fields, and its check ignoring "Submitted by". CPU only, stdlib only."""
import importlib.util, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("gen_readme", os.path.join(HERE, "..", "bench", "gen_readme.py"))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

for e in ("kapitsa", "nace-infer"):
    assert g.version({"version": "", "engine": e}) == g.DASH, e
    assert g.version({"engine": e}) == g.DASH, e
assert g.version({"version": "617f6a2+dirty", "engine": "kapitsa"}) == "617f6a2+dirty"
assert g.version({"version": "0.7.1rc1", "engine": "omlx"}) == "0.7.1rc1"

try:
    g.row("/x/bench/results/a/b.json", {"when": "2026-10-21T00:00:00", "engine": "kapitsa", "label": "x"}, "a")
except KeyError:
    pass
else:
    raise SystemExit("a record without model_name must raise KeyError")

# a prefill size the harness marked INVALID is shown as a dash (with its TTFT), the other size still shows
pts = [{"target": 4000, "prefill_tps": 250.0, "ttft_s": 16.7}, {"target": 16000, "prefill_tps": 244.0, "ttft_s": 68.0}]
assert g.prefill({"prefill": pts}, 4000)["prefill_tps"] == 250.0
r = {"prefill": pts, "invalid": ["prefill_4000: prompt_tokens 4181 outside ±1% of 4133", "cached_16000: follow-up"]}
assert g.prefill(r, 4000) == {}
assert g.prefill(r, 16000)["prefill_tps"] == 244.0
rec = {"when": "2026-10-23T01:29:55", "engine": "mlx-lm", "label": "x", "model_name": "Qwen3.5-4B", "display": "mlx-lm",
       "version": "mlx_lm 0.32.2", "provenance": {"system_start": {"cpu": "Apple M3 Pro", "memsize": str(36 << 30)}},
       **r}
cells = g.row("/x/bench/results/a/b.json", rec, "a")
assert cells[9:12] == (g.DASH, "244", g.DASH), cells[9:12]

a = "| 2026-10-21 | jdoe | M5 Pro, 48 GB | Qwen3.5 4B | 4-bit | Kapitsa | 617f6a2 | 1 |\n"
b = "| 2026-10-21 | John Doe | M5 Pro, 48 GB | Qwen3.5 4B | 4-bit | Kapitsa | 617f6a2 | 1 |\n"
c = "| 2026-10-21 | jdoe | M5 Pro, 48 GB | Qwen3.5 4B | 4-bit | Kapitsa | 617f6a2 | 2 |\n"
assert g.unsubmitted(a) == g.unsubmitted(b)
assert g.unsubmitted(a) != g.unsubmitted(c)
print("test_gen_readme: ok")
