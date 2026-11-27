# Affine MTP sidecar admission

Reference: `b171f98363abd5dfd8ae5d24f786ca77b3801a4a`.
Candidate source: `8e12c3d09ac4118775755000d75f5fdbe6cbca35`.
Runs: MacBook Pro, M4 Pro (20 GPU cores), 48 GB, macOS 15.7.1 (24G231), Apple Clang 17, Release `-O3`.
The runs used the candidate working diff before its commit; the binaries retained the `b171f983` version label.
`source-identity.json` identifies the committed source by hash and records compiler flags, environment and sidecar digests.
An additional explicit configure/build after the source commit passed and reports `kapitsa 0.2.31 (8e12c3d0)`;
that version-only rebuild does not replace the recorded earlier runs. The latest main at preparation, `a2d3ab30`,
adds benchmark documentation/data only.

## Reproduced failure

A safetensors scalar is valid at the file-format level. An affine MTP linear requires a matrix or stacked experts.
The old sidecar path copies tensors and indexes their shapes before checking this distinction.

| Invocation | Reference | Candidate |
| --- | --- | --- |
| Real `load_mtp`, scalar `mtp.fc.weight` | SIGSEGV, exit -11 | Named shape/dtype error, exit 1 |
| Full CLI with the real 4B trunk and scalar-sidecar overlay | SIGSEGV, exit -11 | Named shape/dtype error, exit 1 |
| `own_qlin`, rank-one/row mismatch/bias mismatch/F16 packed weight/zero rows | Accepted | Rejected |
| Valid dense and stacked affine matrices | Accepted | Accepted |

See `probe-{base,candidate}-results.json` and `cli-scalar-model{,-base}.{json,log}` for each command and exit.
The full CLI loaded the existing 4B trunk before reaching the malformed sidecar. It did not serve an inference request.
The model overlay used symlinks and replaced only `mtp.safetensors`; original model files were unchanged.
The large zero-extent overflow fixture already returned a format error in the reference Release run: no overflow crash is claimed.

The initial reference probe linked the complete original library built before edits. The later reference CLI and real-sidecar
comparison use the same complete build with the original `qwen35_spec.cpp` and `affine_pack.h` compiled separately and linked
ahead of the archive. `reference-build.json` records that command. It is a loader comparison, not a separately measured full-model baseline.

## Valid loader output

`27b-sidecar-comparison.json` contains the two executions of `sidecar-compare.cpp` against the existing
`qwen3.8-27b-4bit/mtp.safetensors`. All eight affine matrices passed through the actual `own_qlin` helper.
The complete resulting weight, scale and bias buffers total 238,878,720 bytes in each arm and compare byte for byte.
Both SHA256 values are `4873ecff109b5355a8ece11c9734661335770746d4ef0d03fae6c487403b573f`.
These bytes are loaded tensors, not logits or generated tokens. No speed measurement is included.

The final test command was:

```sh
MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 build/load_error_test --mtp-gpu
```

It takes the GPU lock internally, creates small Metal buffers and dispatches no kernel. Its 13 malformed fixtures reach
the real `load_mtp` and require the named error. Its 30 valid cases cover 3/4/5/6/8-bit weights, dense/stacked layouts and
BF16/F16/F32 parameters through `own_qlin`, checking patterned packed bytes, shapes and converted nonzero parameters.
`mtp-loader-test-final.log` records the passing result. `affine_pack_test` adds CPU rank, dimension and arithmetic-boundary tests.

## CPU checks

The named Release targets built successfully; `build-full-summary.log`, `build-loader-final.log` and the build logs record them.
The supplementary command/results map is `supplement/results.json`. Twenty supplementary commands passed; the CPU mode
of `load_error_test` failed its existing OS page-cache assertion. It does not call the modified MTP path in that mode.

| CPU `load_error_test` source | Resident pages before → after, of 4,420 | Result |
| --- | --- | --- |
| Reference b171f983 test source | 3 → 962 | FAIL `after <= before + pages / 10` |
| Candidate | 3 → 1,474 | Same assertion FAIL |

Both failures are retained (`load_error_test-base.log`, `supplement/15-load_error_test.log`); this is not an all-green CPU claim.
The parser and structured CLI checks use the existing 4B tokenizer; the grammar suite reports 308 passed, 0 failed.
The lint log is `prek-final-code.log`; `check-commands.json` records the completed supplemental invocations.
`vk cpu-tests` completed all 21 suites with 0 failed; `cpu-tests-summary.log` and `cpu-tests/` retain the summary
and every transcript. This suite does not include the failing supplementary page-cache test.

## Reproduction and remaining checks

Copy the archived `make-fixtures.py`, `loader-probe.cpp` and `sidecar-compare.cpp` to a fresh scratch directory before using
them; the fixture generator writes beside itself. Build each source revision in its own worktree with
`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`, and explicitly reconfigure when changing revisions.
The JSON transcripts contain the exact executed probe/CLI arguments. The probe and sidecar comparator take the GPU lock
internally. A full `kapitsa serve` reproduction must also hold that lock and set `NI_LOCK_HELD=1` when wrapped by its caller.

Not verified on a GPU for inference. Before Ready, build complete reference/candidate `kapitsa` and `logit_dump` binaries;
run paired HTTP checks, the standing logit gate and idle native A/B/A/B for 4B, 27B and 35B coverage identified by
`required-checks.json`. Use the same prefixes (`logit_dump --from`) and record each native bench arm, its flags and invalid reasons.
Do not treat the current loader-byte comparison as an inference output check. Confirm that the affine MTP sidecar actually
loads for that path, and keep unrelated known failures in the reports. New results belong in a new evidence directory.

This fix covers sidecar `own_qlin` rank/dtype/shape admission. In-shards MTP tensors, numbered-expert stacking and validation
against every architecture dimension are outside this change. No kernel, format support policy, inference default or performance claim changes.
