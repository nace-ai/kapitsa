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

## After #314: the prefill attention on the matrix units (18:49-18:58)

[#314](https://github.com/nace-ai/kapitsa/pull/314) moved Dory's prefill attention to the same Metal 4 tensor ops
(`tensor/nh_dory_attn_t.metal`, on by default on an M5 with macOS 26, `NI_DORY_ATTN_T=0` keeps the simdgroup kernel).
The 16K profile above had put the GEMMs and `dory_attn_prefill` at 46.7% of the step each, the same reason the GEMMs
were the first target. These two runs are the same configs on main `3112b93` (the merged change), built Release in a
clean worktree at that commit, in one hold of the GPU lock:

```
BENCH_MODELS=~/Documents/nace/models KAPITSA_BIN=/tmp/kap-post2/build/kapitsa \
  compare_engines.py configs/dory.json --only dory-kapitsa,dory-kapitsa-perloop --skip tools,agent,sustain
```

Same Mac and macOS as above, on AC power, one Edge and one Slack process open. Four other sessions were queued on the
GPU lock and the host was reading 3.8 GB of swap while both runs measured (other sessions resident), which is why the
neighbour shows up in the flag below: the last-loop run carries the harness's contention flag on `prefill_16000`
(mean 2.3%, Microsoft Edge Helper, a 5-sample busy run up to 32.7% foreign GPU), so its 16K cell is a lower bound; the
per-loop run is flag-free. The per-loop record also
carries two repetition-loop flags in decode at T = 0.6 (distinct-4-gram 0.4095 and 0.4827, 2 of 5 runs); one of them
is the 76.89 tok/s maximum and the median, 71.15, sits below it.

| Result file | Setting | Decode T = 0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak footprint (GB) | Flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `dory-kapitsa-20261022-184939.json` | 2 loops, last-loop KV (its default) | 71.7 (69.4-72.6) | 72.2 (71.6-72.5) | 1,625 / 1,283 | 0.13 / 0.21 | 3.65 + 0.67 mapped | contention (16K prefill) |
| `dory-kapitsa-perloop-20261022-185535.json` | 2 loops, per-loop KV, speculative decoding | 71.2 (70.3-76.9)<sup>1</sup> | 80.3 (79.9-80.5) | 1,656 / 1,311 | 0.14 / 0.21 | 4.09 + 0.68 mapped | repetition loop (decode) |

1. Two of the five sampled T = 0.6 runs looped (distinct-4-gram 0.41 and 0.48); the fastest, 76.89 tok/s, is one of them, so read the maximum as an upper bound. The median, 71.15, is not a loop.

Median of the pair, against the #287 session above: prefill **1,522 -> 1,641 tok/s at 4K** (1.08x) and **1,142 -> 1,297
at 16K** (1.14x), and against the 03:26 pre-#287 runs, 2.66x and 2.39x. The attention kernel alone is 1.24x / 1.22x
(#314's interleaved A/B) and that A/B put the whole prefill at 1.04x / 1.07x on one day, so the 1.08x / 1.14x here is what
the change reads like across two builds a few hours apart, not a measurement of it; the interleaved pair below measures that
same change and reads 1.068x / 1.139x, above #314's own 1.04x / 1.07x, so the whole prefill moved by roughly 1.05-1.14x
depending on the day. Decode
is unchanged inside the runs' spread (70.7/73.1 -> 71.7/71.2 at T = 0.6, 72.5/80.5 -> 72.2/80.3 greedy). The cached
follow-ups read 0.13-0.14 / 0.21 s against 0.10-0.11 / 0.15-0.16, which is this switch: that interleaved pair (M5 Pro,
macOS 26, main's `3112b93` build, two runs per arm) reads 0.105 / 0.159 s of cached follow-up with `NI_DORY_ATTN_T=0`
against 0.124 / 0.219 on the default, so the path pays +19 ms / +60 ms (+18% / +38%) while cold prefill gains 1,558 / 1,156
-> 1,664 / 1,317 tok/s. At 16K the record's 0.21 s is the default arm's own 0.219 s, so that gap is the switch and nothing
else; at 4K the record's 0.13-0.14 s sits a few milliseconds above the arm's 0.124 s, which no run here explains. The reason
is the shape of the kernel: a cached follow-up prefills a short suffix, where 32-token threadgroups and 16-key tiles have
nothing to amortize, and a fallback below some token count would recover it. That fallback is follow-up work, not part of
this measurement. The peak footprint is unchanged: decode 3.65 / 4.09 GB in both sessions, the 16K-prefill peak
4.18 -> 4.19 GiB on last-loop KV and 5.02 -> 4.98 on per-loop. Against the other engines on this Mac
(mlx-lm 1,401 / 1,278, llama.cpp 938 / 951), the flag-free per-loop run reads 1.18x mlx-lm at 4K and 1.03x at 16K,
and 1.77x / 1.38x llama.cpp.
