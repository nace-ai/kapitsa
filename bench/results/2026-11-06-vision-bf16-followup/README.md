# Follow-up validation

Runs: MacBook Pro (16-inch, 2024), Mac16,7, M4 Pro (14 CPU / 20 GPU cores), 48 GiB,
macOS 15.7.1 (24G231), Apple Clang 17, CMake Release (-O3), macOS deployment target 14.
Reference `d65a0afa`, implementation `ae9ca905`. `identity.json` records the tested tree head,
actual build version and library SHA256. Evidence-only commits do not change the implementation.
Both A/B checkouts were explicitly CMake-configured before building; the reference reports d65a0afa
for the range/vision comparisons. Original published records remain unchanged in their original folder.

`cpu/` contains the full 21-command `vk cpu-tests` result and raw per-test logs, supplemental tests,
real 4B tokenizer/parser/structured checks, build/configure, prek and cleanup logs. Additional checks:
qwen4_config_test, test_drex_max.py, qwen4_exp_pack_test.py, and systemone_codes_driver build.
`dflash_accept_test` was invoked with `--cpu`. No GPU validation is implied by these CPU results.
`required-checks.txt` is the path classifier's checklist, not a statement that every listed model ran;
its model-folder auto-detection misses existing folders here. The explicit coverage below is authoritative.

Rebuild with `cmake -S TREE -B TREE/build -DCMAKE_BUILD_TYPE=Release`, then build the named
CPU targets listed in AGENTS.md and run `.dsh/skills/verify-kapitsa/scripts/vk cpu-tests` plus
the supplemental commands in AGENTS.md. Use `prek run --all-files` after adding new material.

GPU correctness: native `logit_dump MODEL OUT` reference, candidate `--from REFERENCE_OUT`, then
`python3 tools/logit_compare.py REFERENCE_OUT CANDIDATE_OUT --json REPORT` with default floors.
No --reorder, modified floor, or numerical gate override. Each build takes the GPU flock internally.
The 4B run uses defaults, including the 18-unit ANE split and the actual v1 drafter, in a private
symlink overlay of unchanged model files; no user-cache model files were modified.
All 584 rows (8 prefill, 192 batch, 192 step, 192 verify) are bitwise identical and finite.
Verbatim output is `logit-compare.log`; metadata and exact commands are included.

HTTP: reference and this candidate each pass 48/49 standard API checks. Both fail the same queued
keepalive expectation: one heartbeat arrives at about 15 s, then a busy slot frees at about 23 s,
before a second heartbeat. Raw logs are retained. This is not an all-pass HTTP claim.
The server used --slots 2 --context 32768 --disk-cache 0, private localhost, under the GPU lock.

`kapitsa bench --model MODEL --quick --json --out FILE` produced every native record below.
No production NI_/NACE_ overrides were set; KAPITSA_HOME was isolated per run. Each process takes
the native GPU lock and performs its own quiet wait. All records have native invalid=[] and
nominal thermal state, AC/automatic power and zero swapouts. Native repetition flags and other
application GPU activity are still material limitations even when invalid=[]; nothing was discarded.
These text-only native benches do not measure image TTFT. No inference speedup is established.

| Run | Sampled tok/s | Greedy tok/s | 4K TTFT s | Follow-up s | Peak GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| bench-05-base | 205.53 | 202.36 | 2.1232 | 0.0506 | 6.90 |
| bench-06-vision | 194.61 | 207.52 | 2.1029 | 0.0575 | 6.95 |
| bench-07-base | 178.81 | 209.91 | 2.0884 | 0.0574 | 6.94 |
| bench-08-vision | 191.46 | 214.19 | 2.1028 | 0.0561 | 6.94 |

Flags and activity, by run:

```json
[
  {
    "run": "bench-05-base",
    "flags": [],
    "invalid": [],
    "other_gpu_mean_pct": 1.36,
    "other_gpu_max_pct": 5.33,
    "gpu_users": [
      "Google Chrome"
    ]
  },
  {
    "run": "bench-06-vision",
    "flags": [
      "decode_greedy: repetition loop (distinct-4-gram 0.4842)"
    ],
    "invalid": [],
    "other_gpu_mean_pct": 1.0,
    "other_gpu_max_pct": 3.83,
    "gpu_users": [
      "ChatGPT"
    ]
  },
  {
    "run": "bench-07-base",
    "flags": [],
    "invalid": [],
    "other_gpu_mean_pct": 0.61,
    "other_gpu_max_pct": 1.32,
    "gpu_users": [
      "ChatGPT"
    ]
  },
  {
    "run": "bench-08-vision",
    "flags": [],
    "invalid": [],
    "other_gpu_mean_pct": 0.65,
    "other_gpu_max_pct": 3.38,
    "gpu_users": [
      "ChatGPT"
    ]
  }
]
```

Reader: 42 complete expected-byte/guard/padding checks were repeated against the full reference and
candidate libraries. Two A/B pairs for a 64 MiB BF16 tensor have peak RSS 206,340,096 bytes (both A)
and 139,231,232 bytes (both B): exactly 64 MiB lower. Those reader checks overlapped other CPU tests,
so their elapsed times are retained as diagnostics, not a controlled speed claim.

Image towers: vision-reload.cpp links against each full Release library and takes the GPU lock.
It runs generated fixed 64x64, 224x224 and 384x256 RGB inputs through complete 4B/9B towers,
then unloads and reloads them. All 12 old/new full outputs are nonempty, finite/nonzero,
guarded and byte equal; reloads are byte equal. Per-arm SHA256s and raw diagnostics are retained.
Compile the driver with the repository vit_check target's flags/frameworks, then run
DRIVER MODEL EMPTY_OUTPUT_DIRECTORY for each arm. Compare every output file, not sampled elements.
These encoder runs also overlapped CPU tests: load time/RSS are diagnostic, not model speed claims.

Image HTTP: paired repeat passes 18/18 on each build, using Python 3.12.14 / Pillow 13.0.0.
The initial reference image invocation failed before executing tests because fixtures had not yet
been copied; its log is retained. The initial candidate image invocation already passed 18/18.
The paired repeat removes that setup asymmetry. All owned servers exited after validation.

Not verified: broader 27B/35B image towers and model checks, other image architectures/Macs,
controlled image TTFT or whole-model peak memory. Observed Chrome/ChatGPT GPU activity means
the native text benchmark series is not accepted as idle-GPU performance evidence. Further
fixed-image comparisons and paired measurements are pending; the PR remains Draft.
