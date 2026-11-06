# Follow-up validation

Runs: MacBook Pro (16-inch, 2024), Mac16,7, M4 Pro (14 CPU / 20 GPU cores), 48 GiB,
macOS 15.7.1 (24G231), Apple Clang 17, CMake Release (-O3), macOS deployment target 14.
Reference `d65a0afa`, implementation `279ce236`. `identity.json` records the tested tree head,
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
| bench-01-base | 190.60 | 212.17 | 2.0999 | 0.0587 | 6.87 |
| bench-02-range | 188.37 | 204.36 | 2.1007 | 0.0566 | 6.91 |
| bench-03-base | 204.15 | 225.94 | 2.1004 | 0.0578 | 6.92 |
| bench-04-range | 211.48 | 183.86 | 2.0983 | 0.0571 | 6.91 |

Flags and activity, by run:

```json
[
  {
    "run": "bench-01-base",
    "flags": [],
    "invalid": [],
    "other_gpu_mean_pct": 0.0,
    "other_gpu_max_pct": 0.0,
    "gpu_users": []
  },
  {
    "run": "bench-02-range",
    "flags": [],
    "invalid": [],
    "other_gpu_mean_pct": 0.0,
    "other_gpu_max_pct": 0.0,
    "gpu_users": []
  },
  {
    "run": "bench-03-base",
    "flags": [
      "decode: repetition loop (distinct-4-gram 0.4492)",
      "decode: repetition loop (distinct-4-gram 0.4778)"
    ],
    "invalid": [],
    "other_gpu_mean_pct": 0.0,
    "other_gpu_max_pct": 0.0,
    "gpu_users": []
  },
  {
    "run": "bench-04-range",
    "flags": [
      "decode: repetition loop (distinct-4-gram 0.4656)",
      "decode: repetition loop (distinct-4-gram 0.4994)"
    ],
    "invalid": [],
    "other_gpu_mean_pct": 0.0,
    "other_gpu_max_pct": 0.0,
    "gpu_users": []
  }
]
```

CPU quantizer timing: reuse ../2026-11-06-kflash-range-neon/reproduce.py and quant-real.cpp,
extracting the functions from d65a0afa and 279ce236 with identical C++20 -O3 flags. Actual 4B/9B
drafter fc/down/gate tensors, 1/8/all rows, 4/8 bits, both quantizer rounding modes, four A/B pairs;
each full-matrix timing averages three repetitions. Entire W/S/B comparisons precede timing.
The new run starts only after CPU/HTTP/logit jobs finish. Before/after thermal_state=0,
low_power=false and AC power are recorded; private full process inventories are not published.
Full-matrix default rounding saves 45–49% CPU helper time. This is not model-load or decode speed.
Original screening without thermal records is preserved separately and is not substituted for this run.

Actual GPU drafter: 4B and 9B v1 checkpoints, both 4/8-bit, 270 files compared in each of four cases,
zero mismatches (35,198,376 bytes per 4B case; 42,866,088 per 9B case). Final hidden/logit/selector
values were checked finite. `draft-validation.json` records checkpoint/driver/library hashes.
The same qmm_draft_check driver is used in both arms, changing only the explicit constructor bits
from 4 to 0 so existing NI_KFLASH_BITS chooses 4 or 8. This is a harness edit, not a production edit.
Compile draft-quant-check.cpp against each complete libnaceinfer.a with identical Release flags and
the frameworks used by the repository's qmm_draft_check target. It takes the GPU lock internally.
Run DRIVER MODEL prompt.ids OUTPUT and candidate with --from BASE_OUTPUT at each bit width.
The first baseline invocation failed because direct DFlash2::find did not use NI_DFLASH; the retained
failure led to the explicit model/dflash2 symlink overlay used by both arms.

Not verified: broader full-model 9B/27B/35B checks, actual Flash-Next, other Macs. Native greedy
throughput is variable (candidate 183.86–204.36 vs reference 212.17–225.94 tok/s); no no-slowdown
or inference speedup conclusion is drawn. Additional fixed-input full-model runs are pending.
