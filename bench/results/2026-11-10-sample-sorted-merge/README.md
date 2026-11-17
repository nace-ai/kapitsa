# Sorted sampling-list merge

These files are explicitly **redacted derived evidence**, not original execution logs.
The original records and full output files remain private. Each JSON identifies its
source records by relative name and SHA256. Host paths and resource diagnostics are
omitted. `operator-harness.cpp` is the exact source of the executed standalone
operator checker, archived with the measurements; it is not a new production tool.

Runs: Apple M5 Pro, 20 GPU cores, 48 GiB, macOS 26.6.2; Apple clang 17.0.0
(`clang-1700.0.13.5`). Native builds used CMake Release. Reference source:
`0f1db29877966fec6f9208cd57d2e415f13f16f4`; candidate source:
`24f9182b3c9791f71f4478464d1dfb662859a1c3`. `identity.json` records the
separate binaries, kernel sources, harness and model metadata. Adding these evidence
files does not change the measured production source.

## Change and default

For positive-temperature single-row sampling with K from 8 through 32 and at most
128 complete sorted stage-1 lists, one SIMD group merges four list heads per lane.
It advances the winning list instead of reducing the full threadgroup at every rank.
The existing tie order, exhausted-list ID behavior, probability calculation and RNG
step remain covered by old-versus-new checks. Both ordinary and depth-adjusted
stage-2 entry points use the implementation. Other shapes and greedy sampling keep
the existing path.

This is the default for that shape class on every supported Apple GPU. There is no
chip-name gate or opt-in. The two full-model setups below differ in model family;
M3 and M4 were not measured for this change.

## Operator measurements

`operator.json` contains every sample: 135 shapes, six interleaved A/B pairs per
shape, 256 dispatches per timed command buffer, after three A/B warm-up pairs.
The 60 shapes taking the new path use 49.82% to 64.27% less stage-2 GPU time.
Fallback controls range from 1.78% less to 2.02% more time; those samples are retained.

| Sorted lists | K | Reference median (us) | Candidate median (us) | Time change |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 8 | 7.363119 | 3.607503 | -51.01% |
| 64 | 20 | 16.408203 | 7.141846 | -56.47% |
| 64 | 32 | 25.487955 | 10.646891 | -58.23% |
| 122 | 8 | 7.474609 | 3.745117 | -49.90% |
| 122 | 20 | 16.550700 | 7.352132 | -55.58% |
| 122 | 32 | 25.699626 | 10.846923 | -57.79% |

This is an operator result, not a claim of overall inference acceleration.

The standalone checker passed 11,880 small draw pairs with both
`MTL_SHADER_VALIDATION=1` and `MTL_DEBUG_LAYER=1`, then 72,900 expanded pairs.
It checks selected IDs and probabilities against the old kernel, captured top-K
values/IDs against a CPU ordering reference, output guards, ties, signed zero,
infinities, exhausted lists, fallback shapes and depth-adjusted RNG steps including
unsigned wraparound. The embedded production kernel also passed 135 small cases
with both validation flags. An earlier implementation had 51 exhausted-list
failures; it was corrected before the measured production commit. That failure
record remains represented in `validation.json`.

## Complete-model output checks

Qwen3.5-4B 4-bit and NVIDIA Nemotron-3.5-Bolt-74B-A9B-24GB each passed the existing
logit floor comparison: 584 rows per arm, comprising 8 prefill and 192 each of
step, verify and batch. Every row is bitwise identical. An independent audit checked
every float for finiteness and compared the complete `logits.f32`, `records.txt`
and `sequences.txt` bytes in each path. `output-checks.json` retains the comparator
results, gate values, file lengths and hashes. Verify logits are teacher-forced.

Separate actual Qwen generation checks compared all 512 sampled token IDs at
temperature 0.7, K=20, top-p=0.95 and seed 42. Both speculation disabled and
`--spec 1` produced identical complete ID sequences between the two executables,
without EOS and with nonempty text. The latter loaded the native MTP head and ran
293 cycles per arm with 0.74 accepted tokens per cycle; it is a free-generation
MTP check, separate from the teacher-forced logits. See `sampled-tokens.json`.

## HTTP and native measurements

Each model ran 12 fixed HTTP requests: A/B/A/B for K=8, 20 and 32, 512 output
tokens each, temperature 0.7, top-p=0.95, seed 42, speculation disabled. Complete
request JSON and content-plus-reasoning JSON bytes matched across all four arms
within each K. Throughput uses the first and last streamed chunks rather than
individual token timestamps. `qwen-http.json` and `bolt-http.json` retain every run.

| Model | K=8 throughput change | K=20 | K=32 |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B | -0.179% | -0.045% | +0.712% |
| Nemotron Bolt | +0.319% | -0.018% | -0.068% |

The HTTP suite ran on each model's reference and candidate servers with one slot, an
8192-token context and disk cache disabled. Each returned 51 PASS, 3 FAIL and 1 SKIP. The same three
conversation-retention assertions failed in both arms; the concurrent-slot test
was skipped. This is not an all-checks-pass claim or evidence of a general baseline
defect. The exact failure summaries and test digest are in the HTTP evidence.

Unmodified `kapitsa bench --quick --json` also ran A/B/A/B on both models with its
native MTP defaults. `native-abab.json` retains all three per-run sampled and greedy
throughputs, prefill/follow-up measurements, flags and invalid markers. Its nonce
requests vary between runs, so it is not a fixed-input output comparison.

| Model | Median sampled throughput change | Median greedy throughput change |
| --- | ---: | ---: |
| Qwen3.5-4B | -1.212% | -1.973% |
| Nemotron Bolt | +0.702% | +4.133% |

Qwen reference A1/A2 and candidate B1 reported repetition flags; B2 did not. All
four Bolt runs were unflagged. All eight runs had empty invalid lists and passing
tool-call checks. The lower Qwen samples remain in the record. A separate fixed-input
Qwen greedy A/B/A/B control (512 tokens, temperature 0, K=1, speculation disabled)
gave identical complete requests/outputs and +0.0105% median throughput change;
it does not replace or invalidate the original native measurements. No universal
no-regression claim is made.

## Reproduction templates

The following are portable examples, not literal original commands. `$MODEL`,
`$REF_BUILD`, `$BUILD`, `$REF_KERNEL` and `$RESULT` denote paths chosen for the run.
Use a new output location each time. The binaries acquire the exclusive GPU lock;
an external lock holder must supply the existing caller-held marker.

```sh
"$REF_BUILD/logit_dump" "$MODEL" "$RESULT/reference"
"$BUILD/logit_dump" "$MODEL" "$RESULT/candidate" --from "$RESULT/reference"
python3 tools/logit_compare.py "$RESULT/reference" "$RESULT/candidate"
"$BUILD/kapitsa" bench --model "$MODEL" --quick --json --out "$RESULT/native.json"
```

The archived harness can be copied to an unused `tools/sample_merge_probe.cpp`
in a disposable checkout, configured with CMake Release and built as the
`sample_merge_probe` target. It uses the repository's GPU lock. Supply the reference
and candidate `src/kernels/sample.metal` files as its first two arguments:

```sh
MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 \
  "$BUILD/sample_merge_probe" "$REF_KERNEL" src/kernels/sample.metal small
"$BUILD/sample_merge_probe" "$REF_KERNEL" src/kernels/sample.metal check
"$BUILD/sample_merge_probe" "$REF_KERNEL" src/kernels/sample.metal bench
```

The standalone operator binary's host optimization flags were not persisted;
the archived source, binary digest and shader digests identify the executed probe.
The native model executables have separate recorded Release-build identities.

## Other checks and limits

Release builds, all 23 `vk cpu-tests` commands, documentation links, comment and
profile gates, and lint passed. Other chips, other model families, all sampling
parameter combinations, packaged workloads and exhaustive GPU concurrency were
not tested. No model output or throughput number is inferred from operator timing.
