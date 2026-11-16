# SSD prefix cache for Nemotron-H: runs of 2026-11-16

| Folder | Mac | macOS | Model |
| --- | --- | --- | --- |
| `nano-4b-m4/` | Mac mini M4 (10-core GPU), 16 GB | 26.6.2 (25G83) | `nemotron-3-nano-4b-fp8` |
| `nano-9b-m2/` | Mac mini M2 (10-core GPU), 16 GB | 27.2.1 (26B131) | `nemotron-nano-9b-v2-nvfp4` |

**Builds.** The branch's first commit as 414adf55 on main 1846c9ce (`kapitsa`, `logit_dump`) against main 1846c9ce
(`kapitsa-base`, `logit_dump-base`); Release, Xcode Command Line Tools. The branch was rebased afterwards (the same
patch). Its last commit came after these runs and was not run on a GPU: `disk_format()` is no longer cut at 255
bytes and names the Neural Engine split where a model's shapes take it (neither changes the Nano models' string), and
`--help` lists the models.

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
