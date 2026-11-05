# Re-measure of merged #922, #923 and #926 on the M5 Pro 48 GB (5 Nov 2026)

Parent commit against merge commit on this Mac (MacBook Pro M5 Pro, macOS 26.5.2, AC power). Each row says what was timed:
none of these PRs' headline config (Flash-Next 4-bit) can run here, because its pack is about 100 GB and the Mac has 48 GB of RAM and 55 GiB of free disk.

| PR | What was timed | Parent -> merge | Result (speed change; for #926 less time) |
|---|---|---|---|
| #922 | Flint 4-bit 8-row verify | 30.38 -> 29.72 ms step | faster (+2.2%, 3 pairs) |
| #922 | moe4_unique kernel, 3 rows (Flash-Next shape) | 18.48 -> 7.35 us call | faster (+151.7%, 4 pairs) |
| #922 | moe4_unique kernel, 4 rows | 19.58 -> 10.82 us call | faster (+81.0%, 4 pairs) |
| #923 | gate/up kernel, staged vs device-read (off vs on), 3 rows | 168.99 -> 167.23 us call | faster (+1.1%, 4 pairs) |
| #923 | gate/up kernel, 4 rows | 214.92 -> 210.49 us call | faster (+2.1%, 4 pairs) |
| #923 | down kernel, 3 rows | 101.16 -> 96.23 us call | faster (+5.1%, 2 pairs) |
| #923 | down kernel, 4 rows | 131.84 -> 126.02 us call | faster (+4.6%, 2 pairs) |
| #926 | expert chain, T=32 uniform_random (layer-0 EXL3 fixture) | 11.33 -> 11.07 ms | faster (2.3% less time, 2 runs) |
| #926 | expert chain, T=128 uniform_random (layer-0 EXL3 fixture) | 24.09 -> 23.45 ms | faster (2.6% less time, 2 runs) |
| #926 | expert chain, T=512 uniform_random (layer-0 EXL3 fixture) | 27.32 -> 26.62 ms | faster (2.6% less time, 2 runs) |
| #926 | expert chain, T=2048 uniform_random (layer-0 EXL3 fixture) | 55.20 -> 53.62 ms | faster (2.9% less time, 2 runs) |
| #926 | expert chain, T=128 concentrated (layer-0 EXL3 fixture) | 2.61 -> 2.59 ms | about the same (0.7% less time, 2 runs) |
| #926 | expert chain, T=512 concentrated (layer-0 EXL3 fixture) | 9.80 -> 9.62 ms | faster (1.8% less time, 2 runs) |

- #922 (`moe4_unique`) parent `b210e9ab`, merge `f9e85aa7`; #923 (`NI_MOE4_DX`, default off on the M5 Pro) parent `467ff747`, merge `153dff95`, on against off in the merge build; #926 parent `153dff95`, merge `af7d7958` (the PR's own `exl3_register_check`, original and register kernel in one executable, outputs identical: 0 mismatched bytes).
- Legs are not "clean" by the harness bar: its foreign-GPU sampler takes a sample every 2 s and has none on very short legs, some quiet waits were capped at 60 s, and the Mac's load was 2-4. Per-leg labels are in `VERDICT-*.json` (`legs[].gpu`). Pairs where a leg started on battery are dropped.
- `VERDICT-*.json` carry the commit, binary sha256, argv and env of every arm. `verdict.py` writes them from `ab/*/ab.json`; `ab_lc.py` is the interleaved runner (old, new, then new, old), `moe4_time.cpp` the kernel timer (drop it into `tools/` of the parent and merge checkouts to build it), the shell scripts are the exact job scripts.
- MLX on Flint 4-bit (indicative, not a harness leg): mlx-vlm 0.7.11, 893-token prompt, 128 greedy tokens, 3 cold runs, decode 86.1 / 94.1 / 98.6 tok/s (`mlx/`). Kapitsa's own timer on the same model (`flint_bench`, 893-token corpus prompt of the same length but different text, 128 greedy steps, 2 x 3 reps), decode 125.5-138.2 tok/s, median 134.5 (`kapitsa-flint/`): about 43% faster than mlx-vlm's median. The two tools time differently (Kapitsa has no server or Python in the loop), the Mac's load was 4-6, and mlx-vlm needed a weight-type view (`mlx_flint_run.py`: the pack stores packed weights as U8, mlx-vlm wants uint32) and a patched `config.json` (embed_tokens 8-bit), so read it as "faster", not as a headline number. Details in `VERDICT-922.json` (`mlx_summary`).
