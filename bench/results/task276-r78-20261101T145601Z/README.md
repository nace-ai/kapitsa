# R78 attention exchange buffer

A is PR #846 commit `e8ffd2fb`; B is `bde8330c` on branch `task276-attn-exchange-r78`.

All eight interleaved 4K legs exited 0. One warmup and three measured repetitions per leg, with B first in round 2. One outer blocking `lockf -k /tmp/kapitsa-gpu.lock` covered the cohort.

| Model | A median ms | A min/max ms | B median ms | B min/max ms | B/A |
|---|---:|---:|---:|---:|---:|
| Drex Max v1 | 13405.87 | 12709.04 / 13894.62 | 12937.15 | 12462.94 / 13148.85 | 0.96504 |
| Qwen3.8-27B control | 11130.99 | 10754.06 / 11483.60 | 11243.98 | 10790.23 / 11530.00 | 1.01015 |

Max has a 3.50% lower pooled B median, with both rounds favoring B. Qwen neutral (+1.0%, within range). Its causal path is unchanged, and the overlapping observed ranges support neither a win nor a regression claim.

The unchanged e8ffd2fb reference was 15.3% slower for Max and 15.8% slower for Qwen than in R77; R78 followed #209's reported pressure-4 abort and recorded 2821 to 3125 MiB swap versus R77's 192 MiB, with aggregate CPU at 6.56% to 27.43%, but no per-process CPU, swap-I/O or thermal evidence identifies the cause, so only the within-run A/B is comparable. The original metadata is unchanged. Recompute its host-sample bounds with the accompanying comparison script and reference metadata.

## Paired rounds

| Model / round | A median ms | B median ms | B/A |
|---|---:|---:|---:|
| max / 1 | 13508.97 | 12864.36 | 0.95228 |
| max / 2 | 13397.60 | 13052.92 | 0.97427 |
| qwen / 1 | 10802.01 | 10876.02 | 1.00685 |
| qwen / 2 | 11453.43 | 11481.06 | 1.00241 |

## Correctness and limits

Captured residual/normalized vectors and Max readout logits are exactly equal across all measured A/B repetitions. Qwen has no readout head in this backbone driver. The full-vocabulary output gate and generated continuations remain unrun.

Small A/B checks at T=1,L=17 and T=33,L=65 passed with Metal API and GPU validation enabled. Each shape has matching A/B output bit hashes. Relative error against the float reference was 1.70e-4 and 2.17e-4. Their zero-resolution timing fields are not used as performance evidence.

Pre-leg free memory was 59% to 92%, swap used was 2820.88 to 3125.06 MiB, and recorded CPU load was 6.56% to 27.43%. No safety abort occurred. Absolute R78 timings should not be compared to R77 under a different host state.

## Reproduction details

Runs: Apple M5 Pro, 48 GB, macOS 26.5.2, 2026-11-01. Model packs are Drex Max v1 affine Q4 and Qwen3.8-27B affine Q4. The first measured leg began at 14:59:07Z and the cohort finished at 15:06:06Z.

The provenance records the exact commits, binary digests, environment, argv, and token-file digest. Raw stdout and stderr remain in the local experiment directory, and their SHA256 digests accompany this record. No model weights or raw tensors are committed.

Reproduce the metadata comparison from the repository root:

```sh
r=bench/results/task276-r78-20261101T145601Z
python3 "$r/compare_host_context.py" "$r/reference-r77-meta.txt" "$r/meta.txt"
```

The CPU field is aggregate host usage sampled before each leg, despite the original `outside_cpu_percent` name. It does not identify #209 tests, LibreOffice or WebKit as contributors, and allocated swap is not proof of active paging.

## CPU checks and unrun coverage

The four CPU source/layout tests passed. The pinned Release build matched the reference's 84 compile commands, apart from source and build paths. Repository hooks passed. The author did not rerun the full CPU executable/server suite for this branch.

## Evidence

- [analysis-r78.json](https://github.com/nace-ai/kapitsa/blob/task276-attn-exchange-r78/bench/results/task276-r78-20261101T145601Z/analysis-r78.json)
- [meta.txt](https://github.com/nace-ai/kapitsa/blob/task276-attn-exchange-r78/bench/results/task276-r78-20261101T145601Z/meta.txt)
- [provenance.json](https://github.com/nace-ai/kapitsa/blob/task276-attn-exchange-r78/bench/results/task276-r78-20261101T145601Z/provenance.json)
- [raw-output-sha256.json](https://github.com/nace-ai/kapitsa/blob/task276-attn-exchange-r78/bench/results/task276-r78-20261101T145601Z/raw-output-sha256.json)
- [readout.json](https://github.com/nace-ai/kapitsa/blob/task276-attn-exchange-r78/bench/results/task276-r78-20261101T145601Z/readout.json)
- [repetitions.tsv](https://github.com/nace-ai/kapitsa/blob/task276-attn-exchange-r78/bench/results/task276-r78-20261101T145601Z/repetitions.tsv)
- [status.tsv](https://github.com/nace-ai/kapitsa/blob/task276-attn-exchange-r78/bench/results/task276-r78-20261101T145601Z/status.tsv)
- [validation-status.txt](https://github.com/nace-ai/kapitsa/blob/task276-attn-exchange-r78/bench/results/task276-r78-20261101T145601Z/validation-status.txt)
