# Kapitsa: instructions for coding agents and reviewers

This file is the one source of truth for agents (Claude Code, Cursor, Codex, Devin and the like). `CLAUDE.md` and
`.cursor/BUGBOT.md` point here or repeat the review rules; do not grow a second copy of these rules elsewhere.
Contributor-facing details (build, tests, layout, style) are in [CONTRIBUTING.md](CONTRIBUTING.md); the design is in
[docs/architecture.md](docs/architecture.md), the commands and switches in [docs/usage.md](docs/usage.md).

Kapitsa is a native C++20 inference engine for Apple silicon (Metal kernels, the Neural Engine, no MLX or llama.cpp
runtime). Speed and exact outputs are the product. A change that is faster but not the same tokens is a different
change and has to say so.

## Names: KFlash is ours, DFlash2 is upstream (Alish, 6 Nov 2026, #362)

- **KFlash** names everything Kapitsa built on top of DFlash2: the best-first trees, the budget, the history lane, the
  selector update, and any other draft or verify change. Its knobs are `NI_KFLASH_<NAME>` (for example
  `NI_KFLASH_HIST`) and its log lines start with `KFlash:`.
- **DFlash2** names only the upstream method and the drafter checkpoint (`--dflash`, `NI_DFLASH` for the drafter
  folder, `incoai/...-DFlash2`, the `DFlash2` class that loads it). A new extension never gets a `DFLASH` name.
- The earlier `NI_DFLASH_<NAME>` knobs still work as aliases (`kenv()` in `src/util/log.h` reads `NI_KFLASH_<NAME>` first,
  then `NI_DFLASH_<NAME>`); do not remove them without a deprecation note in `docs/usage.md`.

## State on this Mac is not a privacy concern

Kapitsa is a purely local engine: nothing a request teaches it (a learned selector, a history lane, a cache) leaves
the Mac, and there is no telemetry. State that carries across conversations on the same Mac is therefore not a privacy
reason to leave a speed-up out. The reasons that count are speed, memory and exact outputs. A feature that keeps state
across requests still has to say where the state lives and how to reset it.

## What an agent can and cannot check

- **You can** build, run the CPU tests, run `prek`, read and edit code. Do these before every push.
- **You cannot** know a Metal kernel is right or fast without running it on a GPU, and the Metal sources are embedded
  as text and compiled at run time: a successful build says nothing about them. If you did not run it on a Mac with a
  GPU, say so in the PR ("Not verified on a GPU", in the title too; precedent: PR #170). Never write speedups,
  "bit for bit" or "the same outputs" for something you did not run.
- Do not load a model or touch the GPU unless the task says you may. A run takes the machine's GPU for minutes and
  other people's measurements are on it.
- `.dsh/skills/verify-kapitsa` drives the CLI and server and keeps evidence; its `vk checks` reads the diff
  against `origin/main` and says whether a PR needs a GPU run and on which models.

## Build and test (exact commands)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release          # CMake 3.24+, Xcode Command Line Tools, macOS 14+ target
cmake --build build -j --target kapitsa                  # the CLI only; drop --target for every tool too
```

`all` also builds tools that need a newer macOS SDK than 15.x (`tensor_probe` uses `MTLLanguageVersion4_0`): on an older
SDK build the named targets, as CI does.

CPU tests (no model, no GPU), after building `kapitsa api_test execution_session_test http_stream_test bench_test
update_test prefix_cache_test grammar_test structured_check fn_store_test fn_dflash_test drafter_attach_test tokenizer_test iq_layout_test
exl3_test remote_test cluster_test split_place_test gemm_slices_test ffn_share_test device_profile_test qwen4_config_test nh_quant_q8_test json_file_test serve_options_test events_test systemone systemone_test
systemone_codes_probe_test systemone_codes_driver splash_protocol_test splash_runtime_test splash_discover_test splash_plan_test logit_dump`:

```bash
build/api_test
build/prefix_cache_test
build/fn_store_test                          # the streamed-expert cache (src/model/expert_store.h)
build/fn_dflash_test                         # DFlash2's stream_mean reduction checks (Flash-Next drafts)
build/drafter_attach_test                    # kapitsa import's drafter attach: which shape gets which drafter (offline; --net also fetches the 4B one)
build/tokenizer_test                         # Flint's pre-tokenizer split
build/iq_layout_test                         # GGUF IQ experts in the GPU layout (src/kernels/wfmt_iq.metal)
build/exl3_test                              # the EXL3 reference decoder (src/model/moe_exl3.h), Flash-Next's norm check
build/remote_test                            # a split model's reserve on its other Macs (src/model/remote.h)
build/cluster_test                           # several Macs (src/cluster/): TXT, JSON, routing, kapitsa cluster's table
build/split_place_test                       # where a split model's layers go (src/cluster/place.h)
build/gemm_slices_test                       # a large GEMM's 2048-row slices and their bounds (src/model/gemm_slices.h)
build/device_profile_test                    # chip class and the default table from injected facts (src/engine/device_profile.h)
build/qwen4_config_test                      # what a Qwen 4 config asks that Flash-Next's path does not run (src/model/qwen4_config.h)
build/nh_quant_q8_test                       # Q8 load quantization: code/half-scale bytes, rounding, fallback and guards (CPU only)
build/json_file_test                         # config.json parsed once per file (src/util/json_file.h)
build/affine_pack_test                       # MLX affine pack helpers: bit width from shapes, f16 -> bf16 (src/model/affine_pack.h)
build/ffn_share_test                         # the kernels the split's GPU part runs a dense FFN layer on, and its share (src/model/ffn_share.h)
build/systemone_test                         # System One's pointer heads, readout (Drex Max v1) and projected (Drex v2.0) (src/systemone/systemone.cpp)
build/systemone_codes_probe_test             # controlled option-code regression, built and run explicitly in CI
build/execution_session_test                 # one-shot sessions: start, cancel, terminal order, join (src/engine/execution_session.h)
build/serve_options_test docs/serve-config.schema.json  # kapitsa serve's one validator: flags and --config JSON (src/server/service_config.cpp)
python3 tests/test_serve_schema.py build/serve_options_test  # needs: jsonschema; the schema and the validator agree, --config - survives a restart
build/events_test                            # the server's event bus: replay, resets, slow readers, the log ring (src/server/events.h, activity.h)
python3 tests/test_events_schema.py build/events_test  # needs: jsonschema; every event the bus publishes matches schema/control/event.schema.json
build/http_stream_test                       # HTTP streaming: framing, keepalive, disconnect, meters (src/server/http_stream.cpp)
build/ws_upgrade_test                        # a WebSocket upgrade on /v1/responses gets 426, the rest passes through (src/server/ws_upgrade.h)
build/splash_protocol_test                   # Splash protocol v11 frames against Splash 1.4.0's goldens (src/splash/protocol.h)
build/splash_runtime_test                    # the Splash engine child: start, requests, cancel, death (src/splash/runtime.h)
build/splash_discover_test                   # finding Splash's engine and model on a throwaway tree (src/splash/discover.h)
build/splash_plan_test                       # what Kapitsa adds to `splash serve-native`, and when (src/splash/plan.h)
python3 tests/test_server.py --cli build/kapitsa  # its `kapitsa doctor` cases wait on the GPU: with a model run on it they time out (300 s; 9 Nov 2026, #1113)
python3 tests/test_sidecar.py build/kapitsa
python3 tests/test_bench.py build            # needs: jsonschema; on a real Mac only (CI's VM runners fail it)
python3 tests/test_update.py build
python3 tests/test_community.py build        # needs: jsonschema
python3 tests/test_reasoning.py
python3 tests/test_drex_max.py --cli build   # Drex Max v1: head extraction, tokenizer-only request layouts (no model, no GPU)
python3 tests/test_pr_size.py
python3 tests/test_profile_gates.py          # no new inline chip, core or RAM check outside the profile module
python3 tests/test_comment_gates.py          # no comment or doc line in the repo (not third_party/, bench/results/) gives "measured only here" or "nothing was run" as the reason for a narrow default
python3 tests/test_splash_patches.py --apply   # packaging/splash/: clones Splash and applies the patches; no build
python3 tests/test_logit_compare.py
python3 tests/test_sampled_tokens_harness.py # stand-in ni_run executables and temporary locks; no GPU
python3 tests/test_logit_dump.py build      # its output directory only, before the lock and the model
python3 tools/qwen4_exp_pack_test.py         # needs: numpy; the Flash-Next converter's n-gram table from a config.json
python3 tests/test_structured.py --cli build    # with a model folder's tokenizer when one is downloaded
```

Both `systemone_codes_probe_test` and `systemone_codes_driver` are `EXCLUDE_FROM_ALL` targets. CI explicitly builds
and runs the controlled probe. The driver is for per-change old-versus-new evidence, not a standing differential CI
check. Build it explicitly for that evidence. `vk` has separate hardcoded build and test lists. Editing this file
does not update those lists.

For an option-code change, build candidate and reference drivers with the same compiler flags and the same driver
and tokenizer sources. Replace only the reference's [systemone.cpp](<src/systemone/systemone.cpp>) with the
pre-change source and record its commit.

CMake Release uses `-O3`. The standalone `-O0` example in [test_systemone_codes.py](<tests/test_systemone_codes.py>)
is a CPU correctness build, not the Release configuration. Neither configuration measures inference speed.
Compare all 104 requests:

```bash
python3 tests/test_systemone_codes.py --candidate build/systemone_codes_driver --reference /path/to/reference-driver
python3 tests/test_systemone_codes.py --candidate build/systemone_codes_driver --reference /path/to/reference-driver --tokenizer /approved/drex-max-v1/tokenizer.json
python3 tests/test_systemone_codes_harness.py  # script checks with stand-in driver results, not encoder evidence
```

Without `--tokenizer`, the script creates the three synthetic fixtures and checks their goldens. With `--tokenizer`,
both drivers load the specified existing file unchanged. That mode skips only synthetic goldens. Both modes compare
complete outputs and require the known-valid two-option request to produce nonempty tokens in both drivers.
The script reports accepted and rejected counts per driver, the tokenizer SHA256, its parent `model_dir`, and the
JSON's `model.type`, `ignore_merges`, and `pre_tokenizer`. The logged `model_type` is the tokenizer model type,
not the neural model architecture. The script checks the tokenizer digest after each driver run.

Real-vocabulary evidence for Drex Max requires the actual Drex Max v1 metadata identity. A Drex v2 tokenizer is a
substitute only with evidence that its bytes are identical. A Qwen control is optional. Directory names and a
matching tokenizer type do not establish identity. This CPU coverage does not waive the model output, GPU, or
interleaved speed gates below, and does not establish an inference speedup.

Everything else needs a model folder, often the GPU too (`parser_test MODEL_DIR`, `tests/vision/check_cpu.py`, `logit_dump`,
`batch_check`, `ni_v8check`, `tests/test_server.py URL`, `tests/test_disk_cache.py`, `tests/test_draft.py`,
`tests/test_structured.py URL`, `logit_mask_check`, `split_check`, `tests/test_split.py`, `tests/test_pair.py`,
`tests/test_ane_split_abort.py`, `tests/test_systemone.py`, `tests/test_sampled_tokens.py`): run it only when allowed, and report it as run.

Lint: `uv tool install prek && prek install` once, then `prek run --all-files` (the same hooks CI runs).

## Rules that bite here

**GPU lock.** Anything that loads a model takes the exclusive flock on `$BENCH_GPU_LOCK` (default
`/tmp/kapitsa-gpu.lock`), as `lockf -k -t <ms> <file> <command>` does, and a caller that already holds it says so with
`NI_LOCK_HELD=1` / `NACE_GPU_LOCK_HELD=1`, which every tool that takes the lock, and `kapitsa bench`, read through
`GpuLock::held_by_caller()`. Say it: `flock` is per open file description, so a tool wrapped in `lockf ... <tool>`
that does not is blocked by the lock it already holds. A tool that takes no lock itself (`tools/flint_moe_check.cpp`,
`fn_check.cpp`, `fn_q2_bench.cpp`, `ni_handshake.mm`: their headers say so) is meant to be run under
`lockf -k /tmp/kapitsa-gpu.lock`, and needs no marker. A new script, workflow or test that uses the GPU takes the lock;
a test points `BENCH_GPU_LOCK` at a temporary file. See `src/bench/system.mm` (`GpuLock`), `bench/shortpf/locked.sh`.

**Metal kernels.** A kernel that reads or writes out of bounds can fault the GPU and take the desktop down. Run a new or
changed kernel at small sizes first, with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`, and say that you did in the PR.
Kernel sources live in `src/kernels/` (including `nemotron/`, `dory/`, `tensor/`) and are concatenated into one string
by `cmake/embed_*.cmake`; a new `.metal` file is picked up by the glob.

That layer counts a kernel's static threadgroup memory **twice** on this macOS and compares the sum against the device's
32,768 B threadgroup cap, so a dispatch aborts whenever a kernel's static allocation is above 16,384 B, whatever the
change under review. Measured on 2026-10-22, M5 Pro / macOS 26.5.2 (build 25F84): `attn_prefill11`'s 20,480 B reads as
`total used threadgroupMemoryLength(40960) must be <= 32768` and aborts, while `attn_prefill11_k16x`'s 16,384 B reads as
exactly 32,768 B and runs with the layer on. A full prefill therefore aborts with the layer held on `attn_prefill11g`
(20,480 B static, read as 40,960) and `moe_gate_up3_r2` (16,448 B, read as 32,896) no matter what the PR touches; the
driver's own check agrees with the real limit, which `tools/ni_kstats --limits` reports. So: hold the layer at small
sizes when the changed kernel's static memory is 16,384 B or less, and when it is more say in the PR that the layer's
doubling makes a run impossible, with the kernel's size; a PR is not "unverified" for saying that.

**New features ship on by default. No exceptions.** Every opt-in switch is code to maintain and a setting nobody
tunes. There is no config system for opt-in features, so a feature merges only when it is on by default for the whole
class of hardware or model it targets, earned with the interleaved A/B and the output check below. The PR names that
class in its title or first line and in the "Default for" section of the template.

**Defaults follow the concept, not the measured cell.** A matrix-unit (tensor) optimization is on for every Mac with
matrix units, an ANE path for every Mac with a usable ANE, a MoE path for every MoE model, a long-prompt path for every
prompt above its size. Kapitsa is opinionated on purpose; the recurring config sweep checks that the classes still make
sense.

- **Evidence: at least 2 different setups.** Interleaved A/B plus the output check on at least two setups that differ in
  chip class or model family, when two such setups exist for the class (our M5 Pro plus another model, the EC2 M4 Pro,
  or a contributor's Mac). If only one setup exists for the class, one is enough. The default then applies to the whole
  class. Say in the PR which setups ran, which class they stand for, and which members of the class did not run.
- **Narrow gates are the exception.** An exact chip name, core count, RAM size or single model is allowed only when the
  win is both big (at least 10% faster on the target) and exclusive (measured slower, or broken, on the wider class).
  The profile table row cites that evidence. "Only measured here" is not a reason for a narrow gate.
- **One place.** Every device- or model-dependent default lives in the device profile table
  (`src/engine/device_profile.cpp`, printed by `kapitsa doctor --print-profile`). No new inline `apple_family()`,
  chip-name or core-count checks outside it.
- **The fast profile counts as a default** (6 Nov 2026, #363). Kapitsa has two profiles
  ([docs/profiles.md](docs/profiles.md)): `precise`, the exact defaults, and `fast` (`kapitsa serve --profile fast`,
  `KAPITSA_PROFILE=fast`). It is one setting for every member, not an opt-in switch per feature. A feature that moves
  the outputs beyond the floor's gate is on by default for its class when the fast profile turns it on there: it
  reads its switch through `profile_switch()` (`src/engine/profile.h`), its class is a row in the profile table, and
  its output check shows top-1 flips and mean KL against the precise path no larger than the fastest other engine's
  on the same rows (the bar in docs/profiles.md). The class and evidence rules above apply unchanged. A feature that
  passes the floor's gate belongs in `precise`, not only in `fast`.
- **Off everywhere never merges**, even when the flag-off path is byte-identical to main, the idea is established
  elsewhere, or the PR says "a later PR flips the default". "Off by default" in a title is a NO GO. (5 Nov 2026: #819
  and #944 sat open as off-by-default PRs; #944 measured +7% and +6% on the 35B on an M5 Pro and still left both off.)
- **Not faster anywhere yet:** the PR stays open (a draft is fine) and keeps improving until it wins a default, or it is
  closed. It does not merge opt-in "for later".
- **A class member that is slower or broken** is the exception: the profile table row for it says so and cites the
  measurement.

Here a feature means a kernel path, a decode, drafting or verify strategy, a cache or memory policy, or a new user-facing
flag or environment variable. Three things are not features, so this rule does not apply to them: a kill switch for an
existing default (`NI_X=0`), a debug or profiling switch, and a measuring tool. The output check still gates every
default change.

**Changing speed or numerics.** One change per PR. Measure only on an idle GPU (nothing else running), interleave old
and new builds (A B A B), report every run with the Mac, macOS version and commit. A default changes only after an
interleaved A/B *and* the output check of CONTRIBUTING.md ("Output check"): `build/logit_dump` of the old and the new
build, `tools/logit_compare.py` between them, gated at twice the model's noise floor on this Mac
(`tools/logit_floors/<model>.json`). Put its output in the PR. "The same outputs to the bit" is a claim with evidence
("every row bitwise identical") or it is not made; a change that fails the gate on purpose says it moves outputs.
Never re-measure or edit a floor to let a change pass: `tools/logit_floor.py` re-measures one, in a PR of its own.
One change cannot pass that gate and is still legitimate: a kernel that keeps every product and only reassociates the
K reduction (the experts' GEMM on the M5's tensor units instead of the simdgroup kernels), where half*half products are
exact in fp32 and a different order moves ~0.5% of the sums by one half ulp. Gate it with
`tools/logit_compare.py --reorder` instead, whose bounds are twice the first such change's measured drift, as the floor's
gate is twice its measured floor (a floor file's own `reorder_gate` replaces them for its model: Flint on the M5,
CONTRIBUTING.md), and show in the PR that the diff is a reassociation rather than a precision change: the
per-GEMM check in `tools/bench_moe.cpp` (`vs qrm` columns, every output within one half step of the shipped path's, and
both paths' error the same against a double-precision reference) plus what the temperature-0 continuations did (cold 4K
and 16K prompts: the 35B-A3B's came out identical). That flag is a claim about the diff:
anything that changes *what* is computed stays on the floor's gate.

An A/B of an environment switch through `kapitsa bench` needs its own server: bench starts its server with every `NI_`
variable removed ("Left out of the server, as the harness does"), so `NI_X=1 kapitsa bench` measures the default in
both arms (7 Nov 2026, #372: a Lightning `NI_NH_ATTN_Q8` A/B ran A/A). Start `kapitsa serve --model M --port P --ctx
32768 --slots 1 --disk-cache 0` with the switch set, run `KAPITSA_BENCH_SERVER=P kapitsa bench --model M --quick
--json` against it, and check the server's load log that the switch took effect. Bench then marks the result
`INVALID: server: a server kapitsa bench did not start` and exits 1 with the JSON saved: expected for this use, not a
failed run. A server reading weights from an external drive can take 5 to 10 minutes to answer `/health`.

Historical measurements, known gate exceedances and model coverage are recorded in
[docs/numerical-history.md](docs/numerical-history.md), with links to the public changes. Those records do not
waive the output check for a new change or authorize changing a default.

Requests decoded together can legitimately differ at near-ties from alone (docs/architecture.md); a single-stream path
must not change.

**SSD prefix cache.** Conversations' states stay on users' disks across updates (`src/engine/engine_disk.cpp`). A change
to what a state's bytes mean (KV row order or precision, the recurrent state's layout, the drafter's ring, what a
snapshot holds) that the model's `disk_format()` does not show must bump `kStateFormat` there, or new binaries read old
states wrongly.

**Dependencies.** None without a strong reason: Kapitsa builds with Xcode CLT and CMake alone. `third_party/` is
vendored; do not edit it.

**Python.** Tools, benchmarks and tests are stdlib-first scripts. `ruff` runs as a correctness gate only (`ruff.toml`);
do not reformat files you are not changing.

**Frozen data.** Prompts, token ids, corpora and recorded results under `bench/` are inputs to published numbers. Do not
reformat, regenerate or "clean" them; hooks and editors must not touch them. A results folder that gains a run, a row or
a chart is normal, and only additions are invisible. A commit that also *changes* a file already under `bench/results/` or
`bench/community/results/` says so in its subject and why ("re-measured …", "supersedes …", "erratum: …"): the branch no
longer holds the old copy, so a reader cannot tell a declared erratum from an accident. Transcripts are where that bites
hardest: a run script that tees `transcripts/<label>.out` overwrites that label's frozen transcript when the label is
re-run, so commit the new run under a new label, or say in the subject that the transcript is replaced.

**Docs follow numbers.** When a measured number, a default, a flag or a catalog entry changes, grep the repo for the old
value (README, docs/usage.md, docs/benchmarks.md, `tests/`, the catalog in `src/hub/catalog.cpp`) and update every copy. A catalog change also regenerates
`docs/models.md` (`python3 tools/models_table.py`); CI fails on a stale one.

## Pull requests

- One change per PR. Branch off the current `origin/main`; `main` takes squash merges only, and the title becomes the
  commit: state the change and its measured effect (the repo's titles carry numbers and "(#NNN)").
- A new feature says which class of hardware or model it is the default for, and which 2 setups measured it ("Defaults
  follow the concept" above). A feature that is off on every model and Mac stays an open PR.
- The description says what changed, the measured before/after with a `Runs:` line (Mac, macOS, model), and what was
  checked how. A separate "Not verified" section lists what was not run, above all anything needing a GPU.
- Benchmark submissions carry the `benchmark` label, the results table, `Kapitsa version (commit)`, `Flags:`, and say
  `kapitsa bench` produced them. Results come only as pull requests from a fork, never as issues. A result file and the
  regenerated `bench/community/RESULTS.md` (`python3 tools/community_table.py`) ride in the same pull request: the
  `community-table` check reads the merge commit, so a row without its rewrite is red and, once merged, leaves main
  stale until someone regenerates it. Bringing a bench pull request up to date re-runs the command before the merge.

- PRs of more than ~1000 changed lines (generated data and vendored code excluded) fail the `PR size` check. Split them.
  Only a reviewer overrides it (`size/override` label and a `Size override: <why>` line in the description); an author
  does not label their own PR through it.
- Never merge your own PR and never use admin bypasses. Do not change repository settings, rulesets, secrets or runners.
- Workflows: keep the posture of `community-bench.yml`. `pull_request_target` only when the job never runs PR code
  (checkout the base, read the PR as data); never put PR-controlled text (titles, branch names, file names, bodies) into a
  `run:` line, pass it through `env:`. New GPU-using workflows take the GPU lock. Pin tool versions.
- Do not commit agent memory, plans or personal settings (`*-local*`, `.claude/`, `.cursor/plans`). Do not commit
  `build/`, models or `*.bin`.

## Reviewing

`.cursor/BUGBOT.md` carries the review checklist for Cursor Bugbot, which reads only that file. If you change a rule
here that a reviewer must enforce, change it there too.
