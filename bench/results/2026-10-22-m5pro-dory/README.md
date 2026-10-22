# Results, 2026-10-22: MacBook Pro M5 Pro, 48 GB (Dory, current main)

The Nemotron Dory1.0 4B row of [docs/benchmarks.md](../../../docs/benchmarks.md#nemotron-dory10-4b) is a 2026-10-19
session on 0.1.36, before the M5 prefill kernels. These two runs re-measure the two KV settings on current main (engine
`180a650`, built from the docs branch `fd01fe9`, which differs from it only in docs, harness and tests), written by
[bench/harness](../../harness/README.md): `compare_engines.py configs/dory.json --only dory-kapitsa[-perloop] --skip
tools,agent,sustain`, one config at a time under the GPU lock.

MacBook Pro (16-inch, 2026), `Mac17,8`, M5 Pro, 48 GB, macOS 26.5.2, on AC power, one Edge and one Slack process
open. Another app (Drata Agent Helper) used 7% of the GPU through both prefill phases of both runs, so the prefill
cells are lower bounds and both records are flagged; the decode phases ran with the GPU quiet.

| Result file | Setting | Decode T = 0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak footprint (GB) | Flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `dory-kapitsa-perloop-20261022-032657.json` | 2 loops, per-loop KV, speculative decoding | 72.4 (69.0-74.5) | 80.5 (79.8-82.3) | 618 / 543 | 0.10 / 0.15 | 5.1 + 0.65 mapped | contention (prefill) |
| `dory-kapitsa-20261022-031737.json` | 2 loops, last-loop KV (its default) | 71.7 (70.5-72.3)<sup>1</sup> | 72.5 (71.9-72.6) | 618 / 543 | 0.10 / 0.15 | 4.3 + 0.64 mapped | contention (prefill) |

1. One of the five sampled runs looped (distinct-4-gram 0.49; flagged) and was the fastest, 72.3 tok/s.

## After #287: the prefill GEMMs on the matrix units (14:10-14:50)

[#287](https://github.com/nace-ai/kapitsa/pull/287) put Dory's prefill GEMMs on the M5's Metal 4 tensor ops (`nh_gemm_t_fp4_m64`/`fp4l2_m64`, on by default on an M5 with macOS 26, `NI_DORY_TG=0` keeps the simdgroup kernels). These
four runs are the same two configs on main `74b97d9` (the merged change), built Release in a clean worktree, in two
interleaved sessions a few minutes apart — last-loop KV first then per-loop, and the second session reversed:

```
compare_engines.py configs/dory.json --only dory-kapitsa,dory-kapitsa-perloop --skip tools,agent,sustain            # 14:10, 14:12
compare_engines.py configs/dory.json --only dory-kapitsa-perloop,dory-kapitsa --skip tools,agent,sustain --reverse  # 14:22, 14:30
```

Same Mac and macOS as above, on AC power, one Edge and one Slack process open, a browser tab active in two of the four
runs. Three of the four records carry the harness's contention flag (Edge and Slack Helper, means 2.0%, 9.2% and 7.9%
across decode_greedy and both prefill phases), so the prefill cells are lower bounds; the 14:30 run is flag-free.
The 14:22 per-loop record also carries two repetition-loop flags (2 of its 5 sampled T = 0.6 runs, distinct-4-gram 0.47 and
0.49), and it supplies the 74.0 in the per-loop pair's 73.1 median: read the per-loop decode figures as an upper bound.
The last-loop default's decode has no loop flag.

| Result file | Setting | Decode T = 0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak footprint (GB) | Flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `dory-kapitsa-perloop-20261022-141230.json` | 2 loops, per-loop KV, speculative decoding | 72.2 (67.1-74.1) | 77.2 (74.6-78.1) | 1,496 / 1,123 | 0.10 / 0.16 | 4.09 + 0.68 mapped | contention |
| `dory-kapitsa-perloop-20261022-142212.json` | same, second session | 74.0 (66.8-75.5) | 78.8 | 1,516 / 1,114 | 0.10 / 0.15 | 4.09 + 0.67 mapped | contention |
| `dory-kapitsa-20261022-141012.json` | 2 loops, last-loop KV (its default) | 71.5 (71.3-72.2) | 72.1 (69.0-72.4) | 1,517 / 1,137 | 0.11 / 0.15 | 3.65 + 0.67 mapped | contention |
| `dory-kapitsa-20261022-143010.json` | same, second session | 69.8 (68.3-70.7) | 70.8 (70.5-71.6) | 1,526 / 1,146 | 0.10 / 0.16 | 3.65 + 0.67 mapped | — |

Median of the pair, against the 03:26 runs above: prefill **618 -> 1,522 tok/s at 4K** (2.46x) and **543 -> 1,142 at 16K**
(2.10x); decode is unchanged inside the runs' spread (71.7/72.4 -> 70.7/73.1 at T = 0.6, 72.5/80.5 -> 71.5/78.0 greedy,
the lower greedy reads being the two contended records); the peak footprint is 0.5-1.0 GB smaller because the tensor-path
run differs (3.65 against 4.3 GB on last-loop KV; the cause is not measured here). The 16K cell is the one still short of the competition:
mlx-lm's Dory pull request reads 1,401 / 1,278 tok/s and llama.cpp's NVIDIA fork 938 / 951 on this Mac
([docs/benchmarks.md](../../../docs/benchmarks.md#nemotron-dory10-4b)), so Kapitsa is 1.09x mlx-lm at 4K and 0.89x at
16K, and a one-off profile of the 16K step (not stored in this PR) puts the prefill GEMMs and `dory_attn_prefill` at 46.7% each.
