#!/usr/bin/env python3
"""Pins required_checks.py's classification of sample paths: python3 test_required_checks.py (stdlib, no git)."""
import importlib.util, os, sys

spec = importlib.util.spec_from_file_location("rc", os.path.join(os.path.dirname(__file__), "required_checks.py"))
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

SHARED = rc.SHARED
CASES = [  # path, tier, models (None = no model run; "shared" = the 4B and the 35B-A3B)
    ("docs/usage.md", 0, None),
    (".dsh/skills/verify-kapitsa/SKILL.md", 0, None),
    ("src/api/openai.cpp", 0, None),
    ("src/server/bench_cmd.cpp", 0, None),
    ("src/server/chat.html", 0, None),
    ("src/cluster/place.cpp", 0, None),
    ("src/cluster/discovery.cpp", 0, None),
    ("src/server/main.cpp", 1, "shared"),
    ("src/server/beam.cpp", 1, "shared"),
    ("src/server/model_manager.h", 1, "shared"),
    ("src/cluster/stage.cpp", 1, "shared"),
    ("src/cluster/pipe.cpp", 1, "shared"),
    ("src/cluster/wire.h", 1, "shared"),
    ("src/systemone/systemone.cpp", 1, "shared"),
    ("src/engine/grammar.cpp", 0.5, None),
    ("src/engine/engine.cpp", 1, "shared"),
    ("src/unknown/thing.cpp", 1, "shared"),
    ("src/kernels/tensor/qattn_t_flint.metal", 2, ["flint-4bit"]),
    ("src/kernels/tensor/nh_dory_attn_t.metal", 2, ["nemotron-labs-dory1.0-4b-nvfp4"]),
    ("src/kernels/tensor/nh_v8t.metal", 2, ["nemotron-3.5-lightning-nvfp4", "nemotron-labs-dory1.0-4b-nvfp4"]),
    ("src/kernels/qwen4.metal", 2, ["flash-next-q20"]),
    ("src/kernels/zq2moe.metal", 2, ["flash-next-q20"]),
    ("src/kernels/zqwen4v.metal", 2, ["flash-next-q20", "flint-4bit"]),
    ("src/model/dflash2.cpp", 2, SHARED + ["qwen3.8-27b-4bit", "flash-next-q20"]),
    ("src/model/kflash_budget.cpp", 2, SHARED + ["qwen3.8-27b-4bit", "flash-next-q20"]),
    ("src/model/qwen4_exp.cpp", 2, ["flash-next-q20"]),
]

bad = 0
for path, tier, models in CASES:
    t, m, why = rc.classify(path)
    if (t, m) != (tier, models):
        print(f"FAIL {path}: got tier {t} models {m} ({why}), want tier {tier} models {models}")
        bad += 1
print(f"{len(CASES) - bad}/{len(CASES)} classifications as pinned")
sys.exit(1 if bad else 0)
