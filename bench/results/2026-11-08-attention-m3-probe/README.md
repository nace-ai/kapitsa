# Attention scratch: supplemental M3 probe

The `qattn_t_g8d2s8` candidate compiled and ran on an M3 Ultra with Metal API and shader validation enabled. Normal-mode pipeline storage fell from 24,576 to 16,384 bytes. The numerical check did not pass: the pre-change and candidate kernels both reported a maximum absolute error divided by the FP64 reference maximum (`max_abs / ref_max`) of 1.28 on this small case.

Runs: Apple M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1 (25E253); Release `-O3`, arm64, macOS 14 minimum. No model was loaded. Candidate source: `9e85764416be699578fc0cfc0b977d2ed449dbd6`. Reference kernel source: `065c4e36d611d3767acec1ba45b6a7ce965d420a`.

The reference uses the same Release `bench_attn` object and library as the candidate, replacing only the embedded `qattn_t.metal` text with the pre-change version. `reference-build.json` records the compile/link commands, kernel/embed hashes and binary identities. Candidate binary SHA256: `f2a4cc0c909c033814e4022a9eba61ed78e938e26dd128683577e8e91ade3d3c`; reference binary SHA256: `9cfc681885f53a2d0b0e2853a7d39c4958973df90a9748b96b98a63388138c86`.

## Command and results

Every probe used the exclusive GPU lock and `NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 BENCH_ATTN_REPS=1`:

```text
<candidate-or-reference-bench-attn> --all-rows 8 16 qattn_t_g8d2s8
```

The shape was T=8, L=16, Hq=16, Hkv=2, causal attention. The first candidate probe also set `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`. The normal-mode reference/candidate comparison disabled those two layers.

| Probe | Exit | Pipeline static bytes reported | Finite output / canary | max_abs / ref_max |
| --- | ---: | ---: | --- | ---: |
| Candidate, both validation layers | 0 | 32,768 | pass / pass | 1.28 (fail) |
| Pre-change, normal mode | 0 | 24,576 | pass / pass | 1.28 (fail) |
| Candidate, normal mode | 0 | 16,384 | pass / pass | 1.28 (fail) |

The candidate's 32,768-byte debug result is the layer's doubled accounting of the 16,384-byte normal allocation. The reference was not run with the debug layer; that outcome is not claimed.

Here `err` is the maximum absolute output error normalized by the maximum absolute FP64 reference value, not a per-element relative error. Both normal-mode kernels report absolute FP64 error `2.13369358`, reference maximum `1.66722376`, and the same diagnostic FNV hash `8bdcee9886baf741`. This shows a shared numerical failure in this M3 case; matching printed hashes do not establish complete byte equality or numerical correctness. The benchmark returns 0 for finite, guarded output without enforcing an FP64 error threshold, so these exits are runtime checks rather than numerical passes.

## Limits

This probe establishes compilation, guarded execution and pipeline storage on M3 for one small shape. It does not establish M5 correctness, M5 performance, full-model outputs or broader shape coverage. The cause of the shared M3 numerical failure has not been isolated. The sequential normal-mode runs are diagnostic probes, not a qualified interleaved performance A/B; no speedup is claimed.

Required target validation remains unavailable: small M5/macOS 26 cases covering partial/live groups and causal/bidirectional GQA variants, complete old/new kernel outputs against FP64, interleaved latency, model output/HTTP checks and applicable second-setup coverage. The existing M4 host runs macOS 15.7.1 and cannot compile this Metal 4 path in that environment.

Raw evidence: `remote-output.log`, `remote-state.json`, `comparison-output.log`, `comparison-state.json`, `reference-build.json`. The probes completed and exited; none was restarted for this supplement. Earlier CPU/build/lint checks remain separate evidence. No production source or previously committed evidence was changed.

`evidence.tar.gz` preserves the five original measurement files. `SHA256.json` hashes every archive member; `archive-sha256.txt` hashes the archive. No compiled binaries, models or personal configuration are included.
