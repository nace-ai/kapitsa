# Validation

Historical measurements below used native candidate `7062e9d1105fc92eda34b173aa792ad7b5cb0bb8` against main `4256a5119bd093f6088bd46943aaacf5cc60d48a`. They remain labeled with those commits and original flags. The revised branch is based on main `1a6630765d9252959ca8b4c7253340a551a454a7`; its new hardware gate is described below. Post-rebase checks will be recorded separately.

Runs: Apple M4 Pro (20 GPU cores), 48 GB, macOS 15.7.1 (24G231), Qwen3.8-27B-4bit, AC power. Kapitsa 0.2.24; default Metal arithmetic and unchanged numerical floors. ChatGPT remained running.

## Operator measurement

Ordinary sampled decode reuses `topk_stage1_rows_sg` when `sample_sg_` is enabled and K >= 8. The flag defaults to `Gpu::apple10()` (M5); `NI_QWEN_SAMPLE_SG=1` opts in on older Macs and `0` disables it on any Mac. It is read at model load and also gates pipeline prewarming. M1–M4 keep the original path by default. The earlier M4 Pro measurements used the enabled path; reproducing them now requires the opt-in. Greedy/smaller K, stage 2, RNG and speculative-verifier dispatches are unchanged.

`sample-sg-micro-final.jsonl` retains the complete final session: six interleaved A/B pairs for every measured shape and stage, including small-K controls. For one row, vocabulary 248,320, K=32, isolated warm-buffer stage-1 GPU time fell from median 46.232 to 36.325 µs (21.4%). This is selection time, not complete token latency. Initial sessions [v1](sample-sg-micro-v1.jsonl) and [v2](sample-sg-micro-v2.jsonl) remain available with every original sample.

| Pair (A then B) | Main µs | Candidate µs | Thermal A / B |
|---|---:|---:|---|
| 1 | 46.257 | 35.676 | nominal / nominal |
| 2 | 46.285 | 36.333 | nominal / nominal |
| 3 | 46.124 | 36.317 | nominal / nominal |
| 4 | 46.237 | 35.884 | nominal / nominal |
| 5 | 45.744 | 36.337 | nominal / nominal |
| 6 | 46.228 | 36.578 | nominal / nominal |

## Standard request benchmark

Unmodified `kapitsa bench --quick --json`, A/B/A/B, with a fresh `KAPITSA_HOME` per leg. Default DFlash2 is used; its weights SHA-256 is retained in `dflash-sha256.txt`. These runs do not establish a whole-request speedup. All runs were valid, tools passed, swapouts were zero and contention lists were empty. The four `bench-sample-sg-{A1,B1,A2,B2}.json` files and their stderr records retain individual samples, hardware, commits, flags and conditions in this PR; duplicated stdout is archived.

| Run | T=0.6 tok/s | Greedy tok/s | 4K TTFT s | Follow-up s | Thermal | Other GPU mean % |
|---|---:|---:|---:|---:|---|---:|
| A1 | 59.69 | 65.37 | 10.4748 | 0.2466 | nominal | 0.0 |
| B1 | 60.06 | 66.03 | 10.4638 | 0.2456 | nominal | 0.0 |
| A2 | 55.55 | 64.81 | 10.4785 | 0.2444 | fair | 0.0 |
| B2 | 62.2 | 67.05 | 10.4741 | 0.2373 | fair | 0.0 |

## Sampled single-row decoding

`tools/bench_sample_decode.py` starts the real HTTP server with `--spec 0`, temperature 0.7, top-k 32 and seed 42, waits for ANE readiness, warms up, then measures a 2,957-token prompt and a 512-token answer. This engages the changed dispatch; the default speculative benchmark alone does not. All four outputs were nonempty, reached the length limit, matched by SHA-256 and reported drafter source `none`. No whole-request speedup is claimed.

| Run | Output tokens | Decode tok/s | Prefill ms | Thermal before → after |
|---|---:|---:|---:|---|
| A1 | 512 | 15.578 | 7714.0 | 1 → 1 |
| B1 | 512 | 15.600 | 7700.9 | 1 → 1 |
| A2 | 512 | 15.592 | 7700.7 | 1 → 1 |
| B2 | 512 | 15.614 | 7699.4 | 1 → 1 |

Thermal values use NSProcessInfo: 0 nominal, 1 fair. The [four original JSONs, server logs, prompt and generated text](sampled-plain/) are preserved unchanged. Output SHA-256: `44992aeef137b8b91487c87ed994b894022ffee4e1cbab2a699b3560b482414d`.

## Correctness and checks

- `sample_reduce_test --gpu --small` with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`: 135 cases, zero failures ([log](sample-sg-final-validation.log)); then `--gpu`: 360 cases, zero failures ([log](sample-sg-final-correctness.log)). Checks compare the original kernel, an independent CPU reference, candidate order, sampling results and guards, including tails, ties, NaN/inf and signed zero.
- Default 27B output gate: all 584 rows bitwise identical (8 prefill + 192 step + 192 verify + 192 batch). `sample-sg-logit-compare.txt` retains the complete comparison; [metadata](https://github.com/chainhelen/kapitsa/tree/045d9e0a3ab5d7b0fe59322c901b0c1eed5e3691/bench/results/2026-10-29-m4pro-operators-more/sample-sg/logit-metadata/) is archived.
- All 27 CPU groups passed (`cpu/summary.txt`); the installed-model tokenizer grammar/mask checks and `prek run --all-files` passed. [Full CPU, build and lint logs](https://github.com/chainhelen/kapitsa/tree/045d9e0a3ab5d7b0fe59322c901b0c1eed5e3691/bench/results/2026-10-29-m4pro-operators-more/sample-sg/) are preserved. Named targets were built on the macOS 15 SDK.
- API: 48/49 on candidate and main. The same keepalive fixture ends its holder requests after two tokens, freeing slots before the expected 15-second keepalive. [Candidate](sample-sg-api-tests.txt) and [earlier main control](main-api-control-earlier.txt) include the failure; server logs are alongside them.

## Reproduce

Build main and candidate in separate checkouts. GPU commands take the standard GPU lock; run them one at a time on an idle GPU. The script refuses to overwrite an existing output directory.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target kapitsa sample_reduce_test
MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 build/sample_reduce_test --gpu --small
build/sample_reduce_test --gpu
build/sample_reduce_test --bench
python3 tools/bench_sample_decode.py --reference /path/to/main/build/kapitsa --candidate build/kapitsa --model /path/to/qwen3.8-27b-4bit --candidate-sample-sg 1 --prompt bench/results/2026-10-29-m4pro-operators-more/sample-sg/sampled-plain/prompt.txt --out /tmp/kapitsa-sample-new
```

The committed prompt SHA-256 is `6f004e0a164073a1d04d5d84ce8aea4aed706825b342e3a5c09348d0682c542e`, matching all four HTTP records.

## Not verified

The new Apple10 default and override have not yet been run on the rebased head. This update is pushed before local computation checks; older runs above are not relabeled as new results.

Qwen3.5-4B and Qwen3.6-35B-A3B/MoE (not installed), other Apple chips/macOS versions, Metal-4-only tools and real image end-to-end requests.
