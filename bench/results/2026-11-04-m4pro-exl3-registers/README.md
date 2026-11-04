# EXL3 prefill weights decoded into MMA registers

Runs: Apple M4 Pro, 20 GPU cores, 48 GiB, macOS 15.7.1 (24G231), on AC power.
Reference: `d93297da5f37ba3d96797a4a6c45c6f80940378a` (#920).
Candidate: that commit with this PR's `src/kernels/exl3moe.metal` changes; its SHA-256 is
`a85ca4b3c18378551f9f21ea54a2afe3c664ca5223e035b11d490152efd5716b`.
Release C++ (`-O3`), normal Metal fast math. Original and candidate kernels were compiled into
the same executable and library; the original source was renamed to avoid symbol collisions.

## Scope and result

The prefill GEMMs compute their output fragments transposed (`W^T * X^T`). Each physical lane
decodes the codewords its weight fragment consumes directly into registers. This removes weight
staging in threadgroup memory. The K blocks, successive MMA updates, half casts, Hadamard
transforms and external output layout remain as before.

Weights: all 512 layer-0 experts from `beamster/Qwen3.8-Flash-Next-Sushi-2bpw`, pinned revision
`cb6c85e4`; D=2560, I=640, top-10, K2/window 15. **Activations and routing are synthetic.**
This measures the expert prefill chain: gate/up, middle Hadamard/SwiGLU, down and output transform.
It excludes the router, shared expert and final MoE combine. It is not model throughput.

Four interleaved A/B pairs after warmup per shape and phase. Medians include all four samples
for each arm; reduction is `100 * (1 - candidate / reference)`. The headline is run
`16-real-bench-quiet`: exclusive GPU lock, successful quiet preflight, zero observed foreign or
ambient GPU activity, and all thermal readings nominal. WindowServer is excluded by the existing
`gpu_hogs` monitor. Shader/API validation was off during timing.

| Tokens | Routing | Original chain (ms) | Registers (ms) | Time reduction |
| ---: | --- | ---: | ---: | ---: |
| 32 | uniform random | 13.237 | 12.224 | 7.65% |
| 128 | uniform random | 28.120 | 26.041 | 7.39% |
| 512 | uniform random | 31.936 | 29.669 | 7.10% |
| 2048 | uniform random | 64.367 | 59.932 | 6.89% |
| 128 | concentrated | 2.997 | 2.868 | 4.32% |
| 512 | concentrated | 11.326 | 10.632 | 6.13% |

Gate/up alone takes 4.69-7.59% less GPU time; down takes 6.38-8.38% less. Concentrated routing
chooses the same ten experts for every token. Uniform routing samples ten distinct experts per token.

Per-threadgroup scratch falls from 11,008 to 8,704 bytes. Compiler metadata reports gate/up GPRs
59 -> 57 and down GPRs 52 -> 51, with a zero-byte stack for all four pipelines. At source level,
barriers per 128-input block fall from 16 to 2; these counts are not speedup percentages.

## Checks and complete run record

`runs.json` preserves all 16 experimental runs' metadata, environment, source/binary hashes,
contention samples and stdout, including every timing sample. Local path prefixes are replaced
by `[checkout]` and `[evidence]`; timings and output text are otherwise retained. It also contains
the weight file hashes and derived medians for both valid final-kernel timing runs.

- `06-direct-small`: 118 fragment/chain cases, with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`.
- `07-direct-check`: 336 expanded cases, including mixed rates and 4096-wide inputs.
- `11-real-check`: 12 cases using 16 actual layer-0 experts, T=8/33/128/512, random/concentrated
  routing and zero inputs, with both Metal validation layers enabled.
- `12-real-stock`: `exl3_check --pack` with both validation layers. The new permanent regression
  checks 460,800 weight values against the independent CPU decoder: every even rate 16-64 and
  every window 8-16, zero/all-one/random trellises, and output guards. Existing CPU-reference
  comparisons and deliberately corrupted-tile negative controls also pass.
- All differential cases above, plus the six final real-weight benchmark cases, have bitwise
  equal gate/up, middle and output buffers, finite values and intact guards. These are operator
  checks, not full-model logits or generated tokens.

Excluded timing attempts are retained: `09-direct-bench` had Lark activity; `13-real-bench` and
`14-real-bench-final` had Codex activity. `15-real-bench-minimized` failed the Chrome preflight
and launched no timings. `10-direct-bench-quiet` is the valid random-weight run (5.06-7.41% less
chain time). Runs 01-05 belong to a rejected earlier shuffle prototype, not the submitted kernel;
its valid run 05 was slower and is not included in any claimed improvement.

Submission checks: Release build, all 19 groups in `vk cpu-tests`, the 16 additional CPU groups
from `AGENTS.md` omitted by that helper, and `prek run --all-files` passed. The collector's help
and refusal without a GPU lock were also checked without launching GPU work.

## Reproduce

Use a separate checkout at this PR's commit. `exl3_register_check.cpp` is the exact final A/B driver
used for the real-weight runs (SHA-256 `801f1387cfef122d34aceaaf53ef19879fbed8283e0d16c9dd36e5bb62478667`).
It reuses `exl3_check.cpp` and `ni_compile.cpp`; it is kept here as frozen evidence, outside the normal
tools build. `run.py` is the portable form of the local collector, using the same lock, preflight,
monitor interval and acceptance criteria. The driver takes the GPU lock when run without a wrapper.

Download these original files from the pinned checkpoint into a separate `EXL3_WEIGHTS` directory:
`config.json`, `model.safetensors.index.json`, `model-exl3-L00-gate.safetensors`,
`model-exl3-L00-up.safetensors`, and `model-exl3-L00-down.safetensors`. The three expert banks total
638,979,000 bytes; their SHA-256 digests are in `runs.json`. This is only a layer-0 fixture.

From the checkout root, with `EXL3_WEIGHTS` set to that directory:

```sh
EXL3_EVID=bench/results/2026-11-04-m4pro-exl3-registers
EXL3_OUT=$(mktemp -d)
git show d93297da5f37ba3d96797a4a6c45c6f80940378a:src/kernels/exl3moe.metal > "$EXL3_OUT/reference.metal"
cp "$EXL3_EVID/exl3_register_check.cpp" tools/exl3_register_check.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
.dsh/skills/verify-kapitsa/scripts/vk build exl3_register_check exl3_check

.dsh/skills/verify-kapitsa/scripts/vk gpu run --eta 3 --label "EXL3 small validation" -- \
  python3 "$EXL3_EVID/run.py" small --reference "$EXL3_OUT/reference.metal" --out "$EXL3_OUT/small"
.dsh/skills/verify-kapitsa/scripts/vk gpu run --eta 3 --label "EXL3 real weight validation" -- \
  python3 "$EXL3_EVID/run.py" real --reference "$EXL3_OUT/reference.metal" \
  --weights "$EXL3_WEIGHTS" --out "$EXL3_OUT/real"
.dsh/skills/verify-kapitsa/scripts/vk gpu run --eta 4 --label "EXL3 CPU reference" -- \
  python3 "$EXL3_EVID/run.py" stockreal --reference "$EXL3_OUT/reference.metal" \
  --weights "$EXL3_WEIGHTS" --out "$EXL3_OUT/stockreal"
.dsh/skills/verify-kapitsa/scripts/vk gpu run --eta 3 --label "EXL3 real weight A/B" -- \
  python3 "$EXL3_EVID/run.py" realbench --reference "$EXL3_OUT/reference.metal" \
  --weights "$EXL3_WEIGHTS" --out "$EXL3_OUT/realbench"
```

Modes `check`, `bench` and `resources` reproduce the expanded differential checks, random-weight
timings and compiler resources. Keep every run in a new output directory. The collector refuses an
existing directory and records excluded timing attempts. Only accept timings when
`timing_eligible` is true and every thermal reading is nominal. Remove the temporary driver copy
from `tools/` after rebuilding; it does not belong in the production tools list.

## Not verified

Full Flash-Next model `logit_dump` / `logit_compare.py` output gate, greedy generation, end-to-end
prefill/decode throughput, live server/API behavior, other layers/checkpoints, and other Macs/macOS
versions. These operator measurements do not establish the repository's full-model default gate.
