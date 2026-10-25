#!/usr/bin/env python3
"""What a Kapitsa change has to show before it is called "no slower, same outputs": reads the files a branch changed
against BASE (default origin/main) and prints the verification tier, the checks it requires and the models to run.

usage: required_checks.py [--repo DIR] [--json] [BASE]        e.g. required_checks.py origin/main
       required_checks.py --files a.cpp b.metal ...           classify a given file list instead of a git diff

The tiers (the skill's pr-checks.md explains each):
  0    CPU only: cannot change what runs on the GPU or the tokens       -> CPU tests + prek
  0.5  tokens on the CPU (tokenizer, templates, parsers, grammar)       -> + tokenizer/structured checks with a model folder
  1    host code on the hot path (engine, loader, vision, build flags)  -> + GPU: logit gate bitwise, one speed A/B, server e2e
  2    compute (kernels, models, Metal, Neural Engine)                  -> + Metal validation, logit gate on every model reached
  The tier only says what must be run. Whether the change may move outputs (tier 3: a precision or a different
  computation) is a claim of the PR, not of its paths: such a PR fails the gate on purpose and adds quality numbers.
Stdlib only.
"""
import argparse, fnmatch, json, os, subprocess, sys

# ---- models: floor file stem -> folder name under $NACE_MODELS (this Mac's copies) ----
MODELS = {
    "qwen3.5-4b-4bit": ("Qwen3.5 4B 4-bit (dense, Neural Engine split)", "qwen3.5-4b-4bit-mtp"),
    "qwen3.6-35b-a3b-4bit": ("Qwen3.6 35B-A3B 4-bit (MoE)", "qwen3.6-35b-a3b-4bit"),
    "qwen3.8-27b-4bit": ("Qwen3.8 27B 4-bit (dense)", "qwen3.8-27b-4bit-mtp"),
    "flint-4bit": ("Flint 4-bit", "flint-4bit"),
    "nemotron-labs-dory1.0-4b-nvfp4": ("Nemotron-Labs Dory 4B NVFP4", "nemotron-labs-dory1.0-4b-nvfp4"),
    "flash-next-q20": ("Qwen3.8-Flash-Next Q2_0 (SSD-streamed experts, NI_EXPERT_STORE=24)", "flash-next-q20"),
    "nemotron-3.5-lightning-nvfp4": ("Nemotron 3.5 Lightning NVFP4 (no floor file)", "nemotron-3.5-lightning-nvfp4"),
}
# a diff here can change decode tokens or speed: vk checks prints the decode-leg rules of output-check.md
DECODE_GLOBS = ["src/model/*", "src/engine/spec*", "tools/ni_run*"]
SHARED = ["qwen3.5-4b-4bit", "qwen3.6-35b-a3b-4bit"]  # one dense, one MoE: the floor for code every model runs

# ---- rules: first match wins; (glob, tier, models or None, why) ----
# models None = no model run; "shared" = SHARED; a list = those floors.
R = []
def rule(tier, models, why, *globs):
    for g in globs:
        R.append((g, tier, models, why))

rule(-1, None, "vendored: AGENTS.md forbids editing third_party/", "third_party/*")
rule(0, None, "frozen benchmark data: do not reformat or regenerate (AGENTS.md)", "bench/prompts/*",
     "bench/quality/ids_*", "bench/corpus.txt", "bench/results/*")
rule(0, None, "community results: regenerate bench/README.md with `python3 bench/gen_readme.py` (full clone)",
     "bench/community/*")
rule(0, None, "agent skills and review rules", ".dsh/*", ".cursor/*", ".claude/*")
rule(0, None, "docs", "docs/*", "*.md", "LICENSE*", "NOTICE*")
rule(0, None, "CI workflow: keep community-bench.yml's posture, PR text only through env:", ".github/*")
rule(0, None, "noise floor: changes only in its own re-measuring PR (tools/logit_floor.py), never to pass a change",
     "tools/logit_floors/*")
rule(0, None, "CPU-side tests and Python tools", "tests/test_*.py", "tools/*.py", "bench/*.py", "bench/*/*.py",
     "bench/*.sh", "bench/*/*.sh", "*.toml", ".pre-commit-config.yaml", "install.sh", "scripts/*")
rule(0, None, "CLI, HTTP API, downloads, updates, bench client, cluster discovery and placement: no model path",
     "src/api/*", "src/hub/*", "src/update/*", "src/bench/*", "src/util/*", "src/server/cluster_cmd*",
     "src/server/update_cmd*", "src/server/ui*", "src/server/picker*", "src/server/bench_cmd*", "src/server/chat.html",
     "src/cluster/cluster.*", "src/cluster/discovery*", "src/cluster/place*")
rule(0.5, None, "tokens in or out on the CPU: tokenizer, chat template, output parser, grammar, tool arguments",
     "src/tokenizer/*", "src/engine/grammar.*", "src/engine/output_parser.h", "src/engine/tool_args.h",
     "src/engine/reasoning_memory.h", "src/model/logit_mask.cpp", "third_party/minja/*")
rule(1, "shared", "SSD prefix cache: a change to what a saved state's bytes mean bumps kStateFormat "
     "(src/engine/engine_disk.cpp); run tests/test_disk_cache.py", "src/engine/engine_disk.cpp", "src/engine/disk_*.h")
rule(1, "shared", "server request path (serve, beam search, history prefill and reload, routing, pairing): "
     "also tests/test_server.py URL; beam search changes which tokens come out", "src/server/*")
rule(1, "shared", "split model: a peer runs a stage's layers (also tests/test_split.py, tests/test_pair.py)",
     "src/cluster/*")
rule(1, "shared", "Drex decision model: rows encoded and scored by a pointer head on the backbone (also "
     "tests/test_systemone.py with a Drex folder, e.g. ~/.cache/kapitsa/models/drex-v2.0-q4)", "src/systemone/*")
rule(1, "shared", "engine: scheduling, slots, prefix cache, batching on the hot path", "src/engine/*")
rule(1, "shared", "model loading and memory plan", "src/loader/*")
rule(1, "shared", "vision: also run tests/vision/check_cpu.py with a model folder", "src/vision/*")
rule(1, "shared", "GGUF import: also build/iq_layout_test and one imported model's logit gate", "src/gguf/*")
rule(2, "shared", "build flags or kernel embedding: everything compiled can change", "CMakeLists.txt", "cmake/*")
rule(2, ["flint-4bit"], "Flint model code", "src/model/flint*", "src/kernels/flint*")
rule(2, ["nemotron-labs-dory1.0-4b-nvfp4"], "Dory model code", "src/model/dory*", "src/kernels/dory/*")
rule(2, ["nemotron-3.5-lightning-nvfp4", "nemotron-labs-dory1.0-4b-nvfp4"], "Nemotron-H code and kernels",
     "src/model/nemotron*", "src/model/nh_*", "src/model/dspark*", "src/model/lilicorr*", "src/kernels/nemotron/*")
rule(2, ["flash-next-q20"], "Flash-Next: Q2_0 experts streamed from the SSD", "src/model/qwen4_exp*",
     "src/model/moe_q20*", "src/model/expert_store*", "src/kernels/*q20*", "src/kernels/*qsa*",
     "src/kernels/tensor/qsattn*")
rule(2, ["qwen3.6-35b-a3b-4bit"], "GGUF IQ experts (35B GGUF variants)", "src/model/iq_experts*",
     "src/kernels/wfmt_iq*")
rule(2, SHARED + ["qwen3.8-27b-4bit"], "Qwen3.5/3.6/3.8 model code", "src/model/qwen35*")
rule(2, ["flint-4bit"], "Flint prefill attention on the M5's tensor units", "src/kernels/tensor/qattn_t_flint*")
rule(2, ["nemotron-labs-dory1.0-4b-nvfp4"], "Dory prefill attention on the M5's tensor units",
     "src/kernels/tensor/nh_dory_*")
rule(2, ["nemotron-3.5-lightning-nvfp4", "nemotron-labs-dory1.0-4b-nvfp4"], "Nemotron-H kernels on the M5's tensor "
     "units (nh_x2t is Nemotron 3.5 Bolt's IQ2_XXS expert: no Bolt folder on this Mac, say so)",
     "src/kernels/tensor/nh_*")
rule(2, ["flash-next-q20"], "Flash-Next: hyper-connections, PLE, the sparse-attention indexer, Q2_0 routed experts",
     "src/kernels/qwen4.metal", "src/kernels/zq2moe*")
rule(2, ["flash-next-q20", "flint-4bit"], "verify kernels Flash-Next's and Flint's speculative paths share",
     "src/kernels/zqwen4v*")
rule(2, SHARED + ["qwen3.8-27b-4bit", "flash-next-q20"], "DFlash2 and KFlash drafting: the Qwen family and Flash-Next",
     "src/model/dflash2*", "src/model/kflash*", "src/kernels/kflash*")
rule(2, "shared+", "M5 tensor_ops kernels: engage only on an Apple10 GPU with macOS 26; run on an M5",
     "src/kernels/tensor/*")
rule(2, "shared+", "kernel or model code every model may reach", "src/kernels/*", "src/model/*", "src/metal/*",
     "src/ane/*")

def classify(path):
    for g, tier, models, why in R:
        if fnmatch.fnmatch(path, g):
            return tier, models, why
    return 1, "shared", "unclassified path: treated as host code on the hot path until a rule says otherwise"

def changed_files(repo, base):
    out = subprocess.run(["git", "-C", repo, "diff", "--name-only", f"{base}...HEAD"], capture_output=True, text=True)
    if out.returncode:
        sys.exit(f"git diff {base}...HEAD failed: {out.stderr.strip()}")
    files = out.stdout.split()
    dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "-uall"], capture_output=True, text=True).stdout
    files += [l[3:].split(" -> ")[-1] for l in dirty.splitlines() if l[3:]]
    return sorted(set(files))

def kapitsa_top(d):
    out = subprocess.run(["git", "-C", d, "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    top = out.stdout.strip()
    return top if out.returncode == 0 and os.path.isdir(os.path.join(top, "src", "engine")) else None

def default_repo():
    if os.environ.get("VK_ROOT"):
        return os.environ["VK_ROOT"]
    here = os.path.dirname(os.path.realpath(__file__))
    top = kapitsa_top(os.getcwd()) or kapitsa_top(here)
    if not top:
        sys.exit("required_checks.py: not in a Kapitsa checkout (cd into one, --repo DIR or VK_ROOT)")
    return top

def chip_gen():
    try:
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
        return int(brand.split("Apple M")[1][0])
    except (IndexError, ValueError):
        return None

def floor_file(repo, stem, gen):
    for name in ([f"{stem}.m{gen}.json"] if gen and gen >= 5 else []) + [f"{stem}.json"]:
        if os.path.exists(os.path.join(repo, "tools", "logit_floors", name)):
            return name
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="origin/main")
    ap.add_argument("--repo", default=None, help="checkout to read (default: $VK_ROOT, else the Kapitsa checkout "
                    "around the current directory, else the one holding this script)")
    ap.add_argument("--files", nargs="+")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    repo = os.path.abspath(a.repo or default_repo())
    files = a.files or changed_files(repo, a.base)
    rows = [(f, *classify(f)) for f in files]
    tier = max([t for _, t, _, _ in rows] or [0])
    models = []
    for _, t, m, _ in rows:
        if t < 1 or m is None:
            continue
        for s in (SHARED if m in ("shared", "shared+") else m):
            if s not in models:
                models.append(s)
    gen = chip_gen()
    store = os.environ.get("NACE_MODELS", os.path.expanduser("~/.cache/kapitsa/models"))
    model_rows = []
    for s in models:
        label, folder = MODELS[s]
        path = os.path.join(store, folder)
        model_rows.append({"floor": s, "label": label, "floor_file": floor_file(repo, s, gen),
                           "dir": path if os.path.isdir(path) else None})

    checks = ["cpu: `vk cpu-tests` (AGENTS.md's CPU list) and `prek run --all-files`"]
    if tier >= 0.5:
        checks.append("tokens: build/tokenizer_test, build/structured_check, build/grammar_test, "
                      "`python3 tests/test_structured.py --cli build` with a model folder's tokenizer, "
                      "`build/parser_test MODEL_DIR`")
    if tier >= 1:
        checks.append("gpu e2e: `vk serve <4B dir>` then `python3 tests/test_server.py http://127.0.0.1:8090`")
        checks.append("outputs: logit gate per model below (CONTRIBUTING.md \"Output check\"); a change not meant to "
                      "move outputs shows \"every row bitwise identical\" or passes the gate")
        checks.append("speed: one interleaved A B A B on an idle GPU per model below (`kapitsa bench --quick` of "
                      "`vk ref` and build/), every run reported with Mac, macOS, commit")
    if tier >= 2:
        checks.append("metal: changed kernels at small sizes under MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 "
                      "(static threadgroup memory > 16,384 B: say the layer cannot run it, with the size)")
        checks.append("if the PR means to move outputs (tier 3): it fails the gate and says so with numbers, plus "
                      "ni_eval against bf16 (bench/quality) and, for a default, bench/reasoning math500-quick")
    if any(fnmatch.fnmatch(f, g) for f in files for g in DECODE_GLOBS):
        checks.append("decode legs (references/output-check.md \"Decode legs\"): (a) assert non-empty text and print "
                      "the token count before any md5/ids comparison, else VOID; (b) print the count of 248044/248046 "
                      "(<|endoftext|>/<|im_end|>) in the first N ids and the ids before the first, on a chat-templated "
                      "prompt that gives >=256 tokens before <|im_end|>; (c) a speculative row prints the drafter load "
                      "line (drafter, path, sha), tok/cycle and accepted draft tokens/cycle")
    notes = sorted({why for _, t, _, why in rows if t < 0 or "kStateFormat" in why or "frozen" in why
                    or "floor" in why or "workflow" in why or "regenerate" in why or "unclassified" in why})
    if tier < 0:
        tier = 0
    result = {"tier": tier, "files": [{"path": f, "tier": t, "why": w} for f, t, _, w in rows],
              "checks": checks, "models": model_rows, "notes": notes, "chip_gen": gen}
    if a.json:
        print(json.dumps(result, indent=1))
        return
    print(f"tier {tier:g}: {len(files)} changed file(s)")
    for f, t, _, w in rows:
        print(f"  [{t:g}] {f}  ({w})")
    print("required checks:")
    for c in checks:
        print(f"  - {c}")
    if model_rows:
        print(f"models (this Mac: M{gen}; floor file the gate will use):")
        for m in model_rows:
            ff = m["floor_file"]
            if ff and gen and gen >= 5 and f".m{gen}." not in ff:
                ff += f" (an M4 Pro measurement: no .m{gen} floor yet, so the gate may be loose or tight here)"
            print(f"  - {m['label']}: floor {ff or 'NONE (report old vs new numbers; no gate)'}; "
                  f"{'dir ' + m['dir'] if m['dir'] else 'not downloaded under ' + store}")
    for n in notes:
        print(f"note: {n}")

if __name__ == "__main__":
    main()
