# Bugbot review rules for Kapitsa

Kapitsa is a native C++20 inference engine for Apple silicon: Metal kernels (embedded as source, compiled by the GPU
driver at run time), the Neural Engine, a local server. Speed and exact outputs are the product. Review as someone who
cannot run the GPU: CI only builds and runs CPU tests, so nothing automatic validates a kernel's math, bounds or speed.
This file is self-contained on purpose (Bugbot does not follow links); the long form is AGENTS.md in the repo root.

Only report what is likely a real defect or a broken rule below. Do not comment on formatting, naming or taste, and do
not ask for abstractions, interfaces or configuration the change does not need.

## Metal kernels (src/kernels/**, *.metal, *.h in kernels)

- Out-of-bounds reads or writes: every thread's index must be checked against the buffer extent, or the dispatch grid must
  provably divide it. A fault here can hang the GPU and take down the desktop. Look at tails (sizes not a multiple of the
  tile or simdgroup width), `threadgroup` array sizes against the indices used, and offsets that are computed in `int`
  and can overflow for large vocabularies or long contexts (use `uint`/`ulong`).
- Missing `threadgroup_barrier` between a write and a read of threadgroup memory; `simd_*` operations inside divergent code.
- Accumulation precision: reductions (softmax, norm, attention) must accumulate in float; flag a silent change to half
  accumulation or a changed reduction order in a path whose outputs are claimed to be unchanged.
- A kernel signature changed without every encoder call site (src/**/*.cpp, *.mm) updated: argument index, type and
  buffer order must match, and so must constants passed with `setBytes`.
- A new or changed kernel needs, in the PR description, a statement that it ran at small sizes with
  `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`. If the PR changes a kernel and says nothing about it, flag it. The layer
  counts static threadgroup memory twice on macOS 26.x and aborts above 16,384 B of it (AGENTS.md, with the measurements
  and `tools/ni_kstats --limits`), so a statement that gives the size and says the run is impossible is the expected
  answer there; do not ask for a run the layer cannot finish.

## Claims about speed and outputs

- A PR that says faster must show interleaved A/B numbers (old and new builds alternating, on an otherwise idle GPU) with
  the Mac, macOS version and commit. One run, or only the new build, is not evidence.
- A PR that says "same outputs", "bit for bit" or "lossless" must show an output check: `tools/logit_compare.py`'s
  report on whole `logit_dump` dumps of the old and the new build (every path: the prefill path decides whether the
  stricter decode gate applies), "every row bitwise identical" for a bitwise claim, or PASS against the model's noise
  floor in `tools/logit_floors/` for differences only at near-ties. If the diff changes the order of floating-point
  operations, a quantization, a rounding, a tile shape or a sampling path, treat the claim as unproven and say what
  changed. A report that FAILs the gate means the change moves outputs: the PR must say so.
- `tools/logit_compare.py --reorder` is the gate for a change that keeps every product and only reassociates a K
  reduction, and its bounds are twice that change's measured drift. A PR using it must also show the per-GEMM check that
  separates a reassociation from a precision change: `tools/bench_moe.cpp`'s `vs qrm` columns (every output within one
  half step of the shipped path's) and both paths' error against the double-precision reference. Anything that changes
  what is computed stays on the floor's gate.
  Three defaults are accepted past that rule by the user (2026-10-22 and 2026-10-23), see AGENTS.md: the 35B-A3B's
  routed-expert prefill on the tensor units; its prefill attention `qattn_t_g8s4k16` (`NI_QATTN_G8=0` opts out), 1.10x
  over the 4-bit floor's gate (mean KL 5.6e-4); and that attention with the head dim split over a pair of simdgroups,
  `qattn_t_g8d2s8` (#365, `NI_QATTN_D2=0` opts out), 1.27x (mean KL 6.5e-4) — the floor's re-measure pending in a PR
  of its own. Do not flag those three as a rule break; do flag the next 35B change if it is measured against the stale M4 Pro floor without saying so.
- `tools/logit_floors/*.json` change only through `tools/logit_floor.py` (the file records its build, Mac and date), in
  a PR of their own: flag a floor or gate edited by hand, or loosened in the same PR as a change it lets pass.
- A PR whose title or description says it was not verified on a GPU is fine; flag it only when the diff is a kernel or
  numerics change that is presented as verified without evidence.
- More than one unrelated change in a PR (speed change plus refactor plus docs rewrite): say it should be split.

## GPU lock and benchmarks

- Anything that loads a model or uses the GPU (scripts, tests, workflows, tools) must take the exclusive flock on
  `$BENCH_GPU_LOCK` (default `/tmp/kapitsa-gpu.lock`, i.e. `lockf -k -t <ms> <file> <command>`), unless the caller sets
  `NI_LOCK_HELD=1` / `NACE_GPU_LOCK_HELD=1` because it already holds it. Flag a new GPU-using path that skips it. Tests
  must point `BENCH_GPU_LOCK` at a temporary file, never at the real lock.
- `kapitsa bench compare` (bench/harness/kbench.py) must stay a wrapper over the harness's own scripts: one engine at a
  time under the GPU lock, the same protocol for every engine, nothing installed, downloaded or sent for the user. Flag a
  new network call, a telemetry or upload path, an auto-install, or a "reason" in its report that is not derived from a
  configuration fact or a measured number.
- Benchmark and quality inputs (bench/prompts, bench/quality/ids_*, bench/quality/src, bench/results, bench/community/results)
  are frozen: flag edits to them that are not clearly intended, and any regenerated or reformatted copy.

## Security and workflows (.github/**)

- `pull_request_target` workflows must not check out or run the PR's code (checkout the base branch; read the PR as data).
  `.github/workflows/community-bench.yml` and `pr-size.yml` are the pattern. Flag `actions/checkout` with
  `ref: ${{ github.event.pull_request.head.* }}` under that trigger.
- Never interpolate PR-controlled text (title, body, branch name, file names, commit messages, labels) into a `run:` line
  with `${{ }}`; it must go through `env:`.
- No new secrets, broader `permissions:`, or new third-party actions that are not pinned to a major version or a SHA.
- Release (`.github/workflows/release.yml`, `packaging/release/**`): signing, notarization and the version check
  (`KAPITSA_VERSION` in CMakeLists.txt must equal the tag) must keep working; flag changes that would publish an unsigned
  or unverified binary silently.

## C++ and the server

- Request handling in `src/server/**`: unchecked sizes from the request (context length, max tokens, image size, number
  of messages) that reach an allocation or a GPU buffer; JSON fields used without a type check; paths from the request
  used without confinement; secrets or the API key in logs.
- Lifetime and threading: a GPU command buffer, encoder or buffer released while still in flight; state shared between
  the request threads and the batch loop without the lock the neighbouring code uses; a request that can leave the batch
  loop waiting forever.
- SSD prefix cache: a change to a model's KV caches, recurrent state, checkpoints or drafter ring (their order, precision,
  layout or meaning) must change that model's `disk_format()` or bump `kStateFormat` in `src/engine/engine_disk.cpp`;
  otherwise a new binary restores states an old one wrote with the wrong meaning.
- `kapitsa update` and `install.sh`: a download is used only after its sha256 and signature are verified, and the swap stays
  a single rename with the old binary kept for `--rollback`.
- New dependencies: Kapitsa builds with Xcode Command Line Tools and CMake alone. Flag any new library, package manager
  or build-time download. `third_party/` is vendored: flag edits to it.
- Comments in this repo explain *why*; a change to non-obvious numerics or protocol code that removes its explanation
  deserves a comment.

## Docs and numbers

- A changed default, flag, environment variable, measured number or catalog entry must be updated everywhere it is
  stated (README.md, docs/usage.md, docs/benchmarks.md, docs/architecture.md, tests, `src/hub/catalog.cpp`). Flag a stale
  copy.
- A test that cannot fail for a user-visible reason (asserts the mock, or nothing) is not coverage; flag it.

## Size

- A PR of more than about 1000 changed lines, not counting vendored code, recorded results and generated data, should be
  split (the `PR size` check fails above that). A `size/override` label is for a reviewer to give, not the author.
