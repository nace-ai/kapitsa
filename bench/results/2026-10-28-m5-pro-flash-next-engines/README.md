# M5 Pro, 48 GB: Qwen3.8-Flash-Next Q2_0, four engines, 2026-10-27

**Not verified on other Macs.** One MacBook Pro M5 Pro, 48 GB (`Mac17,8`, macOS 26.5.2), other agent sessions
sharing the machine (see Caveats). Same question as the 2026-10-24 bench, now with Splash in the mix: which engine is
fastest for this model on a 48 GB Mac, and where does Kapitsa's Splash backend stand?

| arm | what ran |
|---|---|
| **K** | `kapitsa serve --engine splash` 0.2.23 at `7e478053` (splash-6-ondemand), Splash 1.4.0 engine behind it |
| **K2** | the same at `ff6151c8` (splash-6-plan, #743 head), interleaved with its own plain-Splash set **S2** |
| **S / S2** | plain Splash 1.4.0 (`python -m server.server`) |
| **N** | Kapitsa's native engine, main `e840651c`, `flash-next-q20` pack, MTP speculative decoding (depth 2), `NI_EXPERT_STORE=24` |
| **L** | llama.cpp `c7474afc` `llama-server -ngl 99 -fa on -c 33280 -np 1` on Splash's Q2_0 GGUF shards (66 GB, mmap) |

K, K2, S and S2 all run with `--expert-memory 16 GiB --idle-release off` (**not** Kapitsa's default memory plan,
which is sized from free memory; that default is untested here). N is pinned to a 24 GB expert store, as in the
2026-10-24 bench (the auto size reached memory pressure 2 there), so N is not an auto-size result either.

## Summary (medians; `n` = measured repetitions; decode counts only runs with at least 250 generated tokens)

| arm | decode T=0 tok/s | decode T=0.6 tok/s | TTFT 1K s | prefill 1K tok/s | TTFT 4K s | prefill 4K tok/s | TTFT 16K s | prefill 16K tok/s | turn-2 TTFT s (4K chat) | system wired GiB (max) | swap growth GB (max) | memory pressure (max) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K `7e478053` | 56.2 (n 6) | 55.1 (n 6) | 1.40 (n 6) | 741 (n 6) | 3.44 (n 6) | 1184 (n 6) | 16.46 (n 6) | 1172 (n 6) | 0.53 (n 6) | 31.5 | 0.00 | 1 |
| S | 57.7 (n 6) | 54.8 (n 6) | 1.46 (n 6) | 711 (n 6) | 3.51 (n 6) | 1161 (n 6) | 16.32 (n 6) | 1182 (n 6) | 0.53 (n 6) | 31.5 | 0.00 | 1 |
| K2 `ff6151c8` | 55.9 (n 10) | 52.1 (n 10) | 1.45 (n 10) | 715 (n 10) | 3.45 (n 10) | 1181 (n 10) | 16.54 (n 10) | 1166 (n 10) | 0.56 (n 10) | 31.6 | 0.00 | 1 |
| S2 (paired with K2) | 55.5 (n 10) | 54.0 (n 10) | 1.47 (n 10) | 706 (n 10) | 3.55 (n 10) | 1149 (n 10) | 17.32 (n 10) | 1114 (n 10) | 0.57 (n 10) | 31.6 | 0.00 | 1 |
| N native + MTP | 53.7 (n 12) | 57.2 (n 12) | 1.41 (n 12) | 733 (n 12) | 4.81 (n 12) | 847 (n 12) | 25.48 (n 9) | 757 (n 9) | 0.26 (n 9) | 36.5 | 4.47 | 4 |
| L llama.cpp | 9.0 (n 2) | 9.0 (n 2) | 28.12 (n 2) | 37 (n 2) | 110.74 (n 2) | 37 (n 2) | 634.24 (n 2) | 30 (n 2) | 3.34 (n 2) | 8.6 | 0.00 | 1 |

n is the number of requests per cell: K and S 6 (3 legs of 2), K2 and S2 10 (5 legs of 2), N 12 (6 legs of 2; 9 for 16K and turn 2 because stopped legs lack them), L 2. Every cell's n is in the table. 1K, 4K and 16K are 1035, 4074 and 19286
prompt tokens by the Splash tokenizer (the files are 4,000, 16,400 and 65,600 characters of the docs corpus). **Memory columns** are system-wide, sampled about every 2 s during the leg (`vm_stat`, `sysctl vm.swapusage`, `kern.memorystatus_vm_pressure_level`), maximum over the legs of the arm: wired memory of the whole Mac (Metal buffers count as wired), swap growth since the leg started, and memory-pressure level (1 normal, 2 warning, 4 critical). They are not per-process. **Process RSS is not reported as a comparison:** `ps` RSS undercounts Splash, whose expert memory is wired Metal memory that RSS does not count (the per-leg `peak_rss_mib` is still in the JSON, and so is a `phys_footprint` series from a later pass of two legs each for K2, S2 and N).

What the numbers support:

- **K, K2 and S are the same speed within noise.** Decode ranges 49 to 62 tok/s inside every cell, wider than any gap
  between K, K2 and S. Prefill and turn-2 TTFT differ by under 5% at 1K and 4K; at 16K K2 is 4 to 5% slower than K and S, and S2
  was 10% slower than S, which points at the machine rather than the build. This bench shows
  neither a Kapitsa speed-up nor a slowdown for the Splash backend, and not a difference between 7e478053 and ff6151c8.
- **Kapitsa serving Splash prefills about 1.4 to 1.5x faster than the native engine at 4K and 16K** (1181 vs 847 tok/s at 4K,
  1166 vs 757 at 16K, K2 and N medians) and level at 1K. Native with MTP decodes at the same speed (53.7 vs 55.9 tok/s at T=0; 57.2 vs
  52.1 at T=0.6, K2 vs N; decode noise is 5 to 10 tok/s), with a much faster turn 2 (0.26 s vs 0.56 s for K2) and a hit of 4,435 of 4,476 prompt tokens against 4,320 of 4,442 on the Splash
  backend.
- **Memory.** Native peaked at 36.5 GiB wired with swap growth up to 4.5 GB and memory-pressure level 4 (three legs stopped by the guard; 16K and turn-2 cells have n 9 against 12). The Splash-based arms (K, K2, S, S2) peaked at 31.5 to 31.6 GiB wired, no swap growth, pressure level 1 in every leg. Splash ran at `--expert-memory 16G`, native at `NI_EXPERT_STORE=24`; neither is the arm's default memory plan, and other sessions used memory on the same Mac. This table makes no other memory claim.
- **No arm is "fastest" on every cell.** Splash-based arms win prefill at long context, native wins turn 2, and decode is a tie.

## Caveats

- **llama.cpp (L) is not a fair number.** 66 GB of GGUF does not fit in 48 GB, so it streams from the SSD through mmap,
  and the same machine had other sessions' work in memory (minimum free+reclaimable 11.3 GiB, 2.5 to 4.4 GB of swap in
  use before each leg started). It decoded at 9 tok/s and prefilled at 30 to 37 tok/s. That is about four times slower
  than llama.cpp `0a9ef00` in the 2026-10-24 bench on the same GGUF (39 tok/s decode, 570 tok/s at 4K), so a
  difference in the llama.cpp build, flags (`--jinja` here) or memory state, or all three, is likely. n is 2, L's third leg was
  dropped (each leg takes about 19 minutes) and its 4K and 16K TTFT of 111 s and 634 s are what this run produced, not a
  statement about llama.cpp in general. A cold smoke run (8.5 tok/s, TTFT 4.4 s on 64 tokens) is in `legs/leg0-L.*`, excluded from the table.
- **Not a controlled machine.** Other sessions ran builds and CPU tests between legs. Legs ran one at a time under the GPU
  lock, ABAB-interleaved (K S N L, three passes; K2/S2 afterwards in a separate run of three pairs), so a drift affects the arms alike
  but is not removed. Minimum free+reclaimable memory dropped to 5.4 GiB on K and 2.3 GiB on N. The memory guard
  (abort at pressure level 4, free below 3%, swap above 16 GB) stopped one N leg.
- **Three N legs were stopped at pressure level 4** (`legs/leg10-N`, `leg22-N`, `leg25-N`; a first attempt at leg 10 crashed in the harness and is in
  `aborted/`). Completed cells are in the medians; the 16K cell and the turn 2 of the second repetition are missing from those legs, so N has fewer
  repetitions there (n 9). The N medians include them; dropping them changes no conclusion.
- **K and K2 are different binaries from different times; S and S2 are the same Splash.** K2's S2 set is its own
  control. K vs K2 is not a clean A/B.
- The frozen decode prompt changed once before the first kept leg (an essay prompt stopped at about 130 tokens, so
  "write the numbers 1 to 1000" replaced it, threshold 250 tokens). A first K leg and the first L leg of the old harness
  were discarded and moved to `aborted/` (L's first run was killed by hand after prefill measured 40 tok/s; its output
  is kept).
- Each repetition puts a per-rep nonce at the start of the system message, so nothing is served from a prefix cache
  (`cached 0` on the 1K/4K/16K cells). Turn 2 replays a 4K tool-call chat with the reasoning text dropped on every arm.
  Output text was not compared for equality across arms; decode runs are length-limited at 256 tokens.
- Native N was not run at its auto expert-store size, and Kapitsa/Splash was not run at its default memory plan.
  Earlier Splash-speedups runs saw 68 to 70 tok/s decode with a 16 GiB cache; this run measured 55 to 58 with the same
  setting, so absolute decode numbers here are lower than on a quieter machine (cause not isolated).

## Files

- [scripts/flash_next_bench.py](scripts/flash_next_bench.py): one leg per server start (`leg`) and the summary (`summarize`);
  [scripts/run_all.sh](scripts/run_all.sh) and [scripts/run_k2.sh](scripts/run_k2.sh) are the drivers. Both take the GPU
  lock per leg. Harness first frozen at `6272f582`; later commits fix the decode prompt, a harness crash after a memory
  abort, and L repetitions.
- [prompts/](prompts/): the three frozen prompts. [legs/](legs/): every leg's JSON, harness output and server log, `summary.json`
  (per-arm cells with all values, memory, command, binary version) and `table.md`.
  [aborted/](aborted/): stopped or crashed attempts.
- Reproduce one leg: `lockf -k -t 120 /tmp/kapitsa-gpu.lock python3 scripts/flash_next_bench.py leg --arm K --idx 1 --out legs --reps 2`.
