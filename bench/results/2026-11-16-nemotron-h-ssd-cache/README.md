# SSD prefix cache for Nemotron-H: runs of 2026-11-16

| Folder | Mac | macOS | Model |
| --- | --- | --- | --- |
| `nano-4b-m4/` | Mac mini M4 (10-core GPU), 16 GB | 26.6.2 (25G83) | `nemotron-3-nano-4b-fp8` |
| `nano-9b-m2/` | Mac mini M2 (10-core GPU), 16 GB | 27.2.1 (26B131) | `nemotron-nano-9b-v2-nvfp4` |
| `head-nano-4b-m4/` | the same M4 | 26.6.2 (25G83) | `nemotron-3-nano-4b-fp8`, the branch's head |

**Builds.** The branch's first commit as 414adf55 on main 1846c9ce (`kapitsa`, `logit_dump`) against main 1846c9ce
(`kapitsa-base`, `logit_dump-base`); Release, Xcode Command Line Tools. The branch was rebased afterwards (the same
patch). Its last code commit came after these runs: `disk_format()` is no longer cut at 255 bytes and names the Neural
Engine split where a model's shapes take it, and `--help` lists the models. `head-nano-4b-m4/` is that commit
(`kapitsa`, ebcb6903 before the first commit's subject was reworded: the same tree) against main a5bd06df
(`kapitsa-main`), run with `run2.sh` under one hold of the lock.

**How.** `run.sh` ran every step under one hold of the Mac's GPU lock
(`lockf -k /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 ./run.sh MODEL func logit ttft`).

Per folder:

- `func.log`: `tests/test_disk_cache.py build/kapitsa MODEL --keep` (restarts, write-behind, shared KV blocks, eviction);
  `func-serve.log`: its servers' log.
- `logit_dump-*.log`, `logit_compare.txt` (`tools/logit_compare.py --no-gate`: there is no floor file for the Nano
  models), `logit_cmp.txt` (`cmp` and SHA-256 of every dump file): main against the branch, no cache.
- `ttft-2000.log`, `ttft-8000.log`, `ttft-16000.log`: `tests/test_disk_cache.py --ttft --tokens N --rounds 3`. Each round
  runs turn 2 from the SSD after a restart, then cold on a server without the cache, then from RAM: SSD and cold
  alternate, three pairs. Greedy, thinking off.
- `swap.log`: swap-ins and swap-outs and the two busiest other processes before and after the `func` step and each
  `ttft` size.
- `machine.txt`: macOS, chip, RAM, both binaries' `--version`.
- `head-nano-4b-m4/` only: `ab.log` (`ab.py`, its header says what an arm does) and `ab-serve.log`, its servers' log.

Home and temporary paths are written as `~` and `$TMPDIR`, the Macs' network names and addresses as `<host>` and `<ip>`.

## What the logs say beyond the times

- **Where a hit ends.** Turn 2 is read back up to a restore point before its end and the rest is prefilled, from the
  SSD as from RAM (the same `cached` counts): 29 tokens on the 4B at every size (2,194 of 2,223, 8,686 of 8,715, 16,450
  of 16,479); 90, 81 and 134 on the 9B (2,121 of 2,211, 8,607 of 8,688, 16,396 of 16,530). At the 9B's cold rate
  (114.2 s for 16,530 tokens, 6.9 ms a token) those rows are 0.6 to 0.9 s of its 1.06 to 1.38 s.
- **Sizes.** A restore read 122, 228 and 356 MB on the 4B and 182, 288 and 416 MB on the 9B. Less the cached tokens'
  KV rows at 16,384 B a token (the servers' `16 KB per token`), that leaves about 86 MB on the 4B and 147 MB on the 9B:
  one recurrent state (21 and 27 Mamba-2 layers). The first save of the 16K conversation wrote 441 MB (4B) and 416 MB
  (9B); turn 2 added 174 to 181 MB (4B) and 149 to 155 MB (9B).
- **Cold against the SSD's text.** `turn 2 cold writes the SSD's text: yes` in all 18 rounds. The test prints it and
  does not check it: a cold prefill cuts its chunks at other rows than a follow-up's, which can move a near-tie.
- **Load.** No swap-outs in any step; swap-ins per step (`func`, then the three sizes): 111, 162, 36 and 0 on the M4;
  32, 8, 7 and 46 on the M2. On the M2 `audiomxd` held 71 to 77% of a CPU core in every sample and `configd` 38% in six
  of the eight. On both Macs `XprotectService` was at 46% when `func` began; on the M4 nothing was above 12% after.

## The head on the M4 (`head-nano-4b-m4/`)

- **Namespace.** The cache's folder is `90db0c194e3c837db3d49f0d5e55221f` in `func-serve.log` and `ab-serve.log`, as in
  `nano-4b-m4/func-serve.log`: the Nano 4B's `disk_format()` string did not change.
- **The same checks.** `func.log` ALL PASSED. Turn 2, lowest to highest of three rounds, cold / from the SSD / from RAM:
  2,223 tokens 4.656-4.666 / 0.158-0.161 / 0.111 s; 8,715 tokens 18.929-18.933 / 0.196-0.198 / 0.124-0.126 s; 16,479
  tokens 38.360-38.525 / 0.245-0.246 / 0.138-0.139 s. The SSD's reply is the RAM path's text in all 9 rounds, and the
  cold one's.
- **Requests the cache has nothing for** (`ab.log`): main a5bd06df against the head, both started without a
  `--disk-cache` option, three pairs, main first. Two requests per server, 15 s apart, each after a short warm-up.

  | | Main, rounds 1 / 2 / 3 | Head, rounds 1 / 2 / 3 |
  | --- | ---: | ---: |
  | Request 1 (3,982 tokens): first token | 8.340 / 8.338 / 8.343 s | 8.345 / 8.338 / 8.343 s |
  | Request 1: decode, 157 tokens | 25.0 / 25.1 / 25.1 tok/s | 25.1 / 24.9 / 25.1 tok/s |
  | Request 2 (3,997 tokens): first token | 8.380 / 8.341 / 8.348 s | 8.346 / 8.338 / 8.381 s |
  | Request 2: decode, 142 tokens | 26.1 / 26.1 / 26.1 tok/s | 26.1 / 26.2 / 26.2 tok/s |

  Both arms wrote the same text in every round. The head's cache wrote request 1's conversation in the 15 s between
  the requests (239 MB in 114 to 116 ms) and request 2's as the server stopped (151 MB in 45 to 50 ms).
- **Load.** No swap-outs; swap-ins per step (`func`, the three sizes, `ab`): 0, 2, 0, 8 and 0. `XprotectService` was at
  61% when `func` began and `launchd` at 44% when it ended; nothing above 18% after.
