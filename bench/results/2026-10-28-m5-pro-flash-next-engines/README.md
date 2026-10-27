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

| arm | decode T=0 tok/s | decode T=0.6 tok/s | TTFT 1K s | prompt tokens / TTFT 1K (tok/s) | TTFT 4K s | prompt tokens / TTFT 4K (tok/s) | TTFT 16K s | prompt tokens / TTFT 16K (tok/s) | turn-2 TTFT s (4K chat) | peak footprint GiB (serving process tree) | system used GiB at that peak | system wired GiB (max) | swap growth GB (max) | memory pressure (max) | peak ps RSS MiB (not a memory comparison) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K `7e478053` | 56.2 (n 6) | 55.1 (n 6) | 1.40 (n 6) | 741 (n 6) | 3.44 (n 6) | 1184 (n 6) | 16.46 (n 6) | 1172 (n 6) | 0.53 (n 6) | not measured | not measured | 31.5 | 0.00 | 1 | 2229 |
| S | 57.7 (n 6) | 54.8 (n 6) | 1.46 (n 6) | 711 (n 6) | 3.51 (n 6) | 1161 (n 6) | 16.32 (n 6) | 1182 (n 6) | 0.53 (n 6) | not measured | not measured | 31.5 | 0.00 | 1 | 2938 |
| K2 `ff6151c8` | 55.9 (n 10) | 52.1 (n 10) | 1.45 (n 10) | 715 (n 10) | 3.45 (n 10) | 1181 (n 10) | 16.54 (n 10) | 1166 (n 10) | 0.56 (n 10) | 25.0 (2 legs) | 39.9 | 31.6 | 0.00 | 1 | 2385 |
| S2 (paired with K2) | 55.5 (n 10) | 54.0 (n 10) | 1.47 (n 10) | 706 (n 10) | 3.55 (n 10) | 1149 (n 10) | 17.32 (n 10) | 1114 (n 10) | 0.57 (n 10) | 25.4 (2 legs) | 40.0 | 31.6 | 0.00 | 1 | 2964 |
| N native + MTP | 53.7 (n 12) | 57.2 (n 12) | 1.41 (n 12) | 733 (n 12) | 4.81 (n 12) | 847 (n 12) | 25.48 (n 9) | 757 (n 9) | 0.26 (n 9) | 31.7 (2 legs) | 45.7 | 36.5 | 4.47 | 4 | 32433 |
| L llama.cpp | 9.0 (n 2) | 9.0 (n 2) | 28.12 (n 2) | 37 (n 2) | 110.74 (n 2) | 37 (n 2) | 634.24 (n 2) | 30 (n 2) | 3.34 (n 2) | not measured | not measured | 8.6 | 0.00 | 1 | 32984 |

"Prompt tokens / TTFT" is uncached prompt tokens divided by time to first token. It includes the fixed per-request cost, so it is not pure prefill throughput; that shows at 1K, where the arms reach about 710 to 740 tok/s against about 1,150 to 1,185 at 4K.

n is the number of requests per cell: K and S 6 (3 legs of 2), K2 and S2 10 (5 legs of 2), N 12 (6 legs of 2; 9 for 16K and turn 2 because stopped legs lack them), L 2. Every cell's n is in the table. 1K, 4K and 16K are 1035, 4074 and 19286
prompt tokens by the Splash tokenizer (the files are 4,000, 16,400 and 65,600 characters of the docs corpus). **Memory columns.** *Peak footprint* is the `phys_footprint` (`footprint`) of the serving process tree (K and K2: kapitsa + the Splash child; S2: the Splash server + engine; N: kapitsa), sampled about every second (67 to 77 samples per leg), and *system used at that peak* is the Mac's memory in use (48 GB minus free, inactive, speculative and purgeable pages) at that moment. They were added after the first rotation and come from a pass of two legs each for K2, S2 and N (`legs/leg20` to `leg25`); K, S and L are **not measured**. *System wired*, *swap growth* and *pressure* (1 normal, 2 warning, 4 critical) are system-wide (`vm_stat`, `sysctl vm.swapusage`, `kern.memorystatus_vm_pressure_level`), maximum over all legs of the arm. **ps RSS is not a memory comparison:** it does not count the Metal buffers that hold Splash's expert memory, so it shows 2 to 3 GiB for the Splash arms; it is in the table only so nobody reads it as a result.

**Runs:** 2026-10-27, `Mac17,8` (M5 Pro, 48 GB), macOS 26.5.2. One engine at a time under `/tmp/kapitsa-gpu.lock`, one leg (server start, 2 repetitions per cell) at a time, arms rotated ABAB (K S N L, then K2 S2 with N for the footprint pass), so drift hits the arms alike. Decode at T=0 (seed 1) and T=0.6, temperature passed explicitly on every request. Versions: Kapitsa 0.2.23 at `7e478053` (K), `ff6151c8` (K2) and `e840651c` (N); Splash 1.4.0 (engine sha256 `20b36bcc051145b4`); llama.cpp `c7474afc`.

**Flags:**
- K, K2: `kapitsa serve --engine splash --splash-binary <Splash 1.4.0 engine> --splash-model <Q2_0 GGUF> --context 32k --private --splash-arg --expert-memory --splash-arg 17179869184 --splash-arg --idle-release --splash-arg off`.
- S, S2: `python -m server.server <model> --model ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0 --binary <engine> --max-context 32768 --expert-memory 17179869184 --idle-release off`.
- N: `kapitsa serve --model flash-next-q20 --context 32k --private`, env `NI_EXPERT_STORE=24`. L: `llama-server -ngl 99 -fa on -c 33280 -np 1 --jinja`.
- **`--verify-rows` was not passed on any Splash arm** (K, K2, S, S2 alike), so the engine default `auto` applies. Each Splash leg's server log says so (`legs/leg*-[KS].log`; `verify_rows_in_engine_log` in `summary.json` is `["auto"]` for all four). The log also warns that with `auto` on this MoE target, greedy and seeded outputs can differ from run to run.
- The exact command line of every leg is in its JSON (`cmd`, `env`).

What the numbers support:

- **K, K2 and S are the same speed within noise.** Decode across the Splash cells ranges 45 to 62 tok/s; inside one cell it spans 2.6 to 14.3 tok/s, while the medians of K, K2, S and S2 differ by only 2.2 tok/s at T=0 and 2.9 at T=0.6. Prompt tokens / TTFT medians of the four Splash sets differ by at most 4.7% at 1K and 3.0% at 4K; at 16K K2 is 0.5% and 1.4% slower than K and S, and S2 was 5.8% slower than S (the same software), which points at the machine rather than the build. This bench shows
  neither a Kapitsa speed-up nor a slowdown for the Splash backend, and not a difference between 7e478053 and ff6151c8.
- **Kapitsa serving Splash gets about 1.4 to 1.5x more prompt tokens / TTFT than the native engine at 4K and 16K** (1181 vs 847 tok/s at 4K,
  1166 vs 757 at 16K, K2 and N medians) and level at 1K. Native with MTP decodes at the same speed (53.7 vs 55.9 tok/s at T=0; 57.2 vs
  52.1 at T=0.6, K2 vs N; decode noise is 5 to 10 tok/s), with a much faster turn 2 (0.26 s vs 0.56 s for K2) and a hit of 4,435 of 4,476 prompt tokens against 4,320 of 4,425 to 4,443 on the Splash
  backend.
- **Memory.** Peak footprint of the serving process tree (two legs each): K2 25.0 GiB, S2 25.4 GiB, native 31.7 GiB; the system used 39.9, 40.0 and 45.7 of 48 GiB at those peaks. System wired peaked at 31.5 to 31.6 GiB on the Splash arms and 36.5 GiB on native. Native hit memory-pressure level 4 and swap growth up to 4.5 GB in three counted legs, `leg10-N`, `leg22-N` and `leg25-N` (a fourth attempt, in `aborted/`, did too), each stopped by the guard late in the leg (16K prefill or turn 2), so N's 16K and turn-2 cells have n 9 against 12; the Splash arms stayed at pressure level 1 with no swap growth in every leg. Splash ran at `--expert-memory 16G` and native at `NI_EXPERT_STORE=24`, neither the arm's default plan, with other sessions using memory on the same Mac.
- **Turn 2 on the Splash arms is 0.56 s here; #743's 0.20 to 0.26 s was not reproduced on this chat** (4,320 of about 4,430 prompt tokens cached, uncached tail about 106 tokens). `tools/splash_ab.py --turn2` uses a different 4K chat (4,064 of about 4,125 cached, tail about 60 tokens) and builds turn 2 from the reply content rather than replaying a tool-call chat with its reasoning dropped. The cause of the gap was not isolated, and a longer uncached tail alone is too small to explain it at about 1,180 tok/s prompt tokens / TTFT. Both Splash arms agree with each other here (0.56 and 0.57 s), and native is 0.26 s on this same chat.
- **No arm is "fastest" on every cell.** Splash-based arms win prompt tokens / TTFT at long context, native wins turn 2, and decode is a tie.

## Caveats

- **llama.cpp (L) is not a fair number.** 66 GB of GGUF does not fit in 48 GB, so it streams from the SSD through mmap,
  and the same machine had other sessions' work in memory (the two L legs started with 2.8 and 9.2 GB of swap already in use, and the second had only 11.3 GiB free+reclaimable at its minimum, which makes it the
  stronger confounder for the 16K TTFT). It decoded at 9 tok/s and reached 30 to 37 tok/s prompt tokens / TTFT. That is about four times slower
  than llama.cpp `0a9ef00` in the 2026-10-24 bench on the same GGUF (39 tok/s decode, 570 tok/s at 4K), so a
  difference in the llama.cpp build, flags (`--jinja` here) or memory state, or all three, is likely. n is 2, L's third leg was
  dropped (each leg takes about 19 minutes) and its 4K and 16K TTFT of 111 s and 634 s are what this run produced, not a
  statement about llama.cpp in general. A cold smoke run (8.5 tok/s, TTFT 4.4 s on 64 tokens) is in `legs/leg0-L.*`, excluded from the table.
- **Not a controlled machine.** Other sessions ran builds and CPU tests between legs. Legs ran one at a time under the GPU
  lock, ABAB-interleaved (K S N L, three passes; K2/S2 afterwards in a separate run of three pairs), so a drift affects the arms alike
  but is not removed. Minimum free+reclaimable memory dropped to 5.4 GiB on K and 2.3 GiB on N. The memory guard
  (abort at pressure level 4, free below 3%, swap above 16 GB) stopped four N legs (three in `legs/`, one in `aborted/`; next bullet).
- **Four N attempts were stopped at pressure level 4**: `legs/leg10-N`, `leg22-N` and `leg25-N` are counted; the first attempt at leg 10 (`aborted/N-leg10-pressure4.*`) also hit pressure 4 and then crashed in the harness, so it is kept out. Completed cells are in the medians; the 16K cell and the turn 2 of the second repetition are missing from those legs, so N has fewer
  repetitions there (n 9). The N medians include them; dropping them changes no conclusion.
- **K and K2 are different binaries from different times; S and S2 are the same Splash.** K2's S2 set is its own
  control. K vs K2 is not a clean A/B.
- The frozen decode prompt changed once before the first kept leg (an essay prompt stopped at about 130 tokens, so
  "write the numbers 1 to 1000" replaced it, threshold 250 tokens). A first K leg and the first L leg of the old harness
  were discarded and moved to `aborted/` (L's first run was killed by hand after prompt tokens / TTFT measured 40 tok/s; its output
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
- [scripts/readme_numbers.py](scripts/readme_numbers.py) recomputes the figures quoted in the prose above from `legs/summary.json` and the leg JSONs.
- [prompts/](prompts/): the three frozen prompts. [legs/](legs/): every leg's JSON, harness output and server log, `summary.json`
  (per-arm cells with all values, memory, command, binary version) and `table.md`.
  [aborted/](aborted/): stopped or crashed attempts.
- Reproduce one leg: `lockf -k -t 120 "${BENCH_GPU_LOCK:-/tmp/kapitsa-gpu.lock}" python3 scripts/flash_next_bench.py leg --arm K --idx 1 --out legs --reps 2`.
