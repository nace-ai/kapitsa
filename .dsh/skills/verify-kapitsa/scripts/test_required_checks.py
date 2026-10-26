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
    ("tools/logit_dump.cpp", 1, "shared"),
    ("tools/batch_check.cpp", 1, "shared"),
    ("tools/backend-benchmark.py", 1, "shared"),
    ("tools/flint_bench.cpp", 1, "shared"),
    ("tools/fn_store_test.cpp", 0, None),
    ("tools/ni_run.cpp", 1, "shared"),
    ("tools/bench_test.cpp", 0, None),
    ("tools/logit_compare.py", 0, None),
    ("tools/tokenizer_test.cpp", 0.5, None),
    ("tools/tokenizer_test2.cpp", 0.5, None),
    ("tests/data/nanbeige43_tokenizer.json", 0.5, None),
    ("tests/nanbeige43_tokenizer_fixture.py", 0.5, None),
    ("tests/tokenizer_fixtures/qwen3.6.json", 0.5, None),
    # the rule is narrow: another fixture under tests/ is not swept in
    ("tests/data/logit_rows.json", 1, "shared"),
    ("tests/data/chat_templates.json", 0.5, None),
    ("tests/chat_templates_fixture.py", 0.5, None),
    ("tools/template_prompts.cpp", 0.5, None),
    # and the template fixture's consumer, plus the other tools/*_test.cpp, are not swept in with it
    ("tools/api_test.cpp", 1, "shared"),
    ("tools/grammar_test.cpp", 1, "shared"),
]

bad = 0
for path, tier, models in CASES:
    t, m, why = rc.classify(path)
    if (t, m) != (tier, models):
        print(f"FAIL {path}: got tier {t} models {m} ({why}), want tier {tier} models {models}")
        bad += 1
print(f"{len(CASES) - bad}/{len(CASES)} classifications as pinned")

# decode legs: a GPU tool whose source prints tok/s, or a tools/ diff that adds a tok/s string; not a plain GPU tool
import tempfile  # noqa: E402
with tempfile.TemporaryDirectory() as repo:
    os.makedirs(os.path.join(repo, "tools"))
    open(os.path.join(repo, "tools/x_bench.cpp"), "w").write('printf("%.1f tok/s\\n", r);')
    open(os.path.join(repo, "tools/logit_dump.cpp"), "w").write("int main() { return 0; }")
    open(os.path.join(repo, "tools/notes.py"), "w").write("print('tok/s')")
    LEGS = [
        ("GPU tool printing tok/s", ["tools/x_bench.cpp"], [], "prints tok/s: tools/x_bench.cpp"),
        ("GPU tool without tok/s", ["tools/logit_dump.cpp"], [], ""),
        ("non-GPU tool printing tok/s", ["tools/notes.py"], [], ""),
        ("tools/ diff adds a tok/s string", ["tools/logit_dump.cpp"], ['  fprintf(stderr, "%g tokens/s", t);'],
         "a tools/ diff adds a tok/s string"),
        ("decode path wins", ["src/model/qwen35.cpp", "tools/x_bench.cpp"], [], "decode path"),
    ]
    for name, files, added, want in LEGS:
        got = rc.decode_legs_why(files, [f for f in files if rc.emits_toks(repo, f)], added)
        if got != want:
            print(f"FAIL decode legs, {name}: got {got!r}, want {want!r}")
            bad += 1
    print(f"{len(LEGS)} decode-leg cases checked")
sys.exit(1 if bad else 0)
