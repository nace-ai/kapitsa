# Results, 2026-11-01: MacBook Pro M5 Max, 128 GB (Kapitsa's cells on main `7f4e709`)

Kapitsa's M5 Max cells re-measured on main `7f4e709` (Kapitsa 0.2.25, Release) at each model's fastest setting on
main, against the cells they replace. Kapitsa only: no other engine ran, and the other engines' cells in
[docs/benchmarks.md](../../../docs/benchmarks.md#m5-max) keep their earlier dates. Written by
[bench/harness](../../harness/README.md), one config at a time under the GPU lock, 04:52-09:36 local time:
`compare_engines.py configs/<model>.json --only <label> --skip tools,agent` for the setting sweeps and
`compare_engines.py configs/<model>.json --only <label> --agentic --skip tools,agent,cache,sustain` for the cells, which
adds the harness's context sweep (cold prefill and decode at 1K, 2K, 4K, 8K, 16K and the largest prompt that fits the
32K context). MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), every run on
AC power in High Power mode ([machine.json](machine.json)).

**How the settings were chosen.** Per model a short sweep, one run per setting: the drafter main has for it and its
depth or block, and the switches docs/usage.md turns on by default or documents as faster for that model. Then three
runs of the fastest setting that passed the harness's checks (five for Super-VL, whose cell came out slower), interleaved
across models. Every model's fastest setting is its existing label, the one its old cell ran, so no config changes. Kept
out: opt-in switches that change outputs (`NI_NH_ATTN_Q8` on Lightning: about 3% faster decoding, "slightly less
accurate"), and the kill switch `NI_KFLASH=0`, which was swept as a check but is neither a default nor documented as
faster (Lightning's finding below).

**A cell** is the median of its runs' medians, with the runs' range. Decode is a 512-token coding answer at T 0.6 (Flint
at its card's T 1.0), five answers per run; prefill is one cold prompt per size per run.

## Old cells against new

| Model | Setting (label) | Old cell | Prefill 4K / 16K, old → new | Change | Decode T 0.6, old → new (range) | Change | Greedy, old → new | Runs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Flash-Next 180B-A7B | MTP head, depth 2 (default) (`flash-kapitsa`) | [2026-10-22, `fd85910`, automatic](../2026-10-22-m5-max/results/flash-kapitsa-20261022-004742.json) | 1,165 / 1,140 → **1,701 / 1,613** | +46.0% / +41.5% | 105.8 → **108.3** (107.7-108.7) | +2.4% | 106.0 → 107.3 (+1.2%) | 3 |
| Nemotron 3.5 Lightning 30B-A3B | DSpark drafter, KFlash (default) (`lightning-kapitsa`) | [2026-10-22, `fd85910`, automatic](../2026-10-22-m5-max/results/lightning-kapitsa-20261022-010104.json) | 5,401 / 5,763 → **5,738 / 6,238** | +6.2% / +8.2% | 267.4 → **287.3** (270.8-295.3) | +7.4% | 277.9 → 301.6 (+8.5%) | 3 |
| Nemotron 3.5 Super-VL 120B-A12B | LiLiCorr drafter, KFlash (default) (`super-kapitsa`) | [2026-10-22, `fd85910`, automatic](../2026-10-22-m5-max/results/super-kapitsa-20261022-010647.json) | 1,117 / 1,192 → **1,132 / 1,189** | +1.4% / -0.2% | 93.7 → **86.8** (79.2-91.6) | -7.4% | 87.2 → 85.8 (-1.6%) | 5 |
| Nemotron Dory1.0 4B | 2 loops, per-loop KV (`--loop-kv per`): prompt lookup, else 1-loop self-drafts (`dory-kapitsa-perloop`) | [2026-10-22, `039ddea`, automatic](../2026-10-22-m5-max/results/dory-kapitsa-perloop-20261022-014626.json) | 2,368 / 1,966 → **3,169 / 2,531** | +33.8% / +28.7% | 119.1 → **122.5** (117.1-123.2) | +2.8% | 129.0 → 133.5 (+3.5%) | 3 |
| Flint 40B-A2.5B (T 1.0) | no speculation (default) (`flint-kapitsa`) | [2026-10-26, `ed2b07d`, High Power](../2026-10-26-m5-max/results/flint-kapitsa-20261026-054727.json) | 5,721 / 5,049 → **5,770 / 5,284** | +0.9% / +4.6% | 267.0 → **264.9** (263.1-266.9) | -0.8% | 268.2 → 268.5 (+0.1%) | 3 |
| Qwen3.6 35B-A3B | DFlash2 drafter, KFlash (default) (`q36-35b-kapitsa`) | [2026-10-25, `cd20dcc`, automatic](../2026-10-25-m5-max/results/q36-35b-kapitsa-20261025-041336.json) | 5,963 / 5,270 → **6,036 / 5,298** | +1.2% / +0.5% | 339.1 → **346.0** (341.4-349.2) | +2.0% | 364.5 → 371.1 (+1.8%) | 3 |
| Qwen3.8 27B | DFlash2 drafter, 16-row block, KFlash (default) (`q38-27b-kapitsa`) | [2026-10-25, `cd20dcc`, High Power](../2026-10-25-m5-max/results/q38-27b-kapitsa-20261025-205253.json) | 907 / 898 → **904 / 880** | -0.3% / -2.0% | 133.2 → **143.3** (142.9-147.7) | +7.6% | 145.0 → 151.8 (+4.7%) | 3 |
| Nemotron 3.5 Bolt 74B-A9B (24 GB) | LiLiCorr drafter, KFlash, GPU + Neural Engine prefill (default) (`bolt-kapitsa`) | [2026-10-25, `cd20dcc`, High Power](../2026-10-25-m5-max/results/bolt-kapitsa-20261025-215417.json) | 1,038 / 1,035 → **1,075 / 1,100** | +3.6% / +6.3% | 159.5 → **154.0** (153.7-161.9) | -3.4% | 169.1 → 168.7 (-0.2%) | 3 |

Every pair of cells is the same label, model folder and harness phase. The old cells ran in automatic power mode
(2026-10-22, 2026-10-25 35B-A3B) or High Power mode (the rest); the new ones all in High Power mode. Slower than the
old cell: Super-VL's decode (-7.4%, five runs; the old one was one contention- and loop-flagged run), Bolt's decode
(-3.4%; the old cell, 159.5, is inside the new runs' 149.9-161.9 counting the sweep run), Flint's decode (-0.8%, the
old 267.0 just above the new runs' 263.1-266.9) and the 27B's prefill (-0.3% / -2.0%). The build is Release on
`7f4e709`, the power mode is High Power, and no decode phase is contention-flagged (the app's load is 2.5% or less in
every decode phase); the extra Super-VL and Bolt runs did not move them.

## Every measured label

| Label | Prefill 4K / 16K, median (range) | Decode T 0.6 | Greedy | Cached follow-up 4K / 16K (s) | Peak to 16K / with the 32K sweep (GiB) | App GPU %, prefill / decode / sweep | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `flash-kapitsa` | 1,701 (1,675-1,721) / 1,613 (1,580-1,654) | 108.3 (107.7-108.7) | 107.3 (107.1-108.8) | 0.09 / 0.08 | 70.6 / 71.2 | 5.5 / 1.7 / 4.9 | contention: prefill_16000, prefill_4000, sweep |
| `flash-kapitsa-plain` | 1,878 (1,869-1,885) / 1,698 (1,688-1,715) | 86.1 (85.6-86.4) | 87.6 (87.6-88.0) | 0.08 / 0.08 | 70.5 / 71.1 | 4.6 / 1.6 / 4.7 | contention: prefill_16000, sweep |
| `lightning-kapitsa` | 5,738 (5,678-5,853) / 6,238 (6,222-6,297) | 287.3 (270.8-295.3) | 301.6 (265.1-305.4) | 0.06 / 0.07 | 19.0 / 19.1 | 6.2 / 2.1 / 6.6 | contention: prefill_16000, prefill_4000, sweep |
| `lightning-kapitsa-chain` | 5,796 (5,459-6,389) / 6,291 (6,200-6,296) | 314.8 (278.4-316.5) | 315.4 (291.6-327.8) | 0.06 / 0.07 | 19.0 / 19.1 | 6.1 / 2.0 / 6.8 | contention: prefill_16000, prefill_4000, sweep |
| `super-kapitsa` | 1,132 (1,091-1,201) / 1,189 (1,141-1,243) | 86.8 (79.2-91.6) | 85.8 (79.5-88.8) | 0.18 / 0.20 | 70.3 / 70.4 | 4.6 / 1.9 / 4.6 | loop; contention: prefill_16000, sweep |
| `dory-kapitsa-perloop` | 3,169 (3,167-3,186) / 2,531 (2,522-2,550) | 122.5 (117.1-123.2) | 133.5 (133.0-133.7) | 0.08 / 0.14 | 5.1 / 6.2 | 8.8 / 1.8 / 11.3 | loop; contention: prefill_16000, prefill_4000, sweep |
| `dory-kapitsa` | 3,184 (3,082-3,189) / 2,496 (2,475-2,520) | 121.6 (121.2-122.1) | 122.4 (121.8-122.4) | 0.08 / 0.13 | 4.3 / 5.2 | 8.2 / 1.6 / 11.7 | loop; contention: prefill_16000, sweep |
| `flint-kapitsa` | 5,770 (5,752-5,789) / 5,284 (5,283-5,285) | 264.9 (263.1-266.9) | 268.5 (268.0-268.6) | 0.07 / 0.08 | 22.9 / 23.5 | 3.8 / 1.9 / 4.1 | cached 16K follow-up empty; contention: sweep |
| `q36-35b-kapitsa` | 6,036 (6,016-6,093) / 5,298 (5,294-5,309) | 346.0 (341.4-349.2) | 371.1 (360.2-378.4) | 0.04 / 0.05 | 22.0 / 22.5 | 5.6 / 1.6 / 4.6 | contention: prefill_16000, prefill_4000, sweep |
| `q38-27b-kapitsa` | 904 (902-915) / 880 (866-883) | 143.3 (142.9-147.7) | 151.8 (148.1-151.9) | 0.13 / 0.14 | 20.1 / 21.4 | 8.2 / 2.5 / 8.6 | contention: prefill_16000, prefill_4000, sweep |
| `bolt-kapitsa` | 1,075 (1,070-1,078) / 1,100 (1,099-1,103) | 154.0 (153.7-161.9) | 168.7 (158.6-173.3) | 0.13 / 0.14 | 22.8 / 22.9 | 5.3 / 1.9 / 5.7 | contention: prefill_16000, prefill_4000, sweep |

*Peak*: the server's `phys_footprint_peak`, through the 16K prefill (comparable with the old cells, which stopped at
16K) and with the 32K sweep. *App GPU %*: the worst phase's mean GPU load from other processes, which here is the
Claude desktop app driving the session (below). *loop*: one or more answers fell into a repetition loop (distinct
4-grams below 0.5), which flatters a speculative decoder. *cached 16K follow-up empty*: Flint, a model that always
reasons, answered one 16K follow-up with no content, so the harness stopped that cached series (its cold 16K prefill
is unaffected).

## From 1K to 32K context

The context sweep in the same runs: two cold requests per size per run, each answering a 192-token question about the
document; the cell is the median over the runs. Its decode rate is that answer's, not the 512-token coding answer's
above, so its 4K point differs from the decode cell. "32K (max)" is the largest prompt that fits the 32K context with the
answer: 31.5-32.0K tokens (29.2K on Flint, whose tokenizer reads the same corpus text in fewer tokens).

| Prefill tok/s | 1K | 2K | 4K | 8K | 16K | 32K (max) | prompt tokens at 32K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `flash-kapitsa` | 1,618 | 1,591 | 1,720 | 1,589 | 1,530 | 1,463 | 31,550 |
| `flash-kapitsa-plain` | 1,703 | 1,716 | 1,852 | 1,621 | 1,561 | 1,485 | 31,550 |
| `lightning-kapitsa` | 5,394 | 6,528 | 6,280 | 5,858 | 5,457 | 4,790 | 32,006 |
| `lightning-kapitsa-chain` | 5,631 | 6,580 | 6,344 | 5,865 | 5,520 | 4,847 | 32,006 |
| `super-kapitsa` | 941 | 1,071 | 1,126 | 1,149 | 1,142 | 1,105 | 32,002 |
| `dory-kapitsa-perloop` | 2,414 | 3,143 | 3,016 | 2,693 | 2,347 | 1,800 | 32,006 |
| `dory-kapitsa` | 2,962 | 3,011 | 2,978 | 2,709 | 2,373 | 1,815 | 32,006 |
| `flint-kapitsa` | 4,280 | 5,601 | 5,726 | 5,405 | 4,939 | 3,972 | 29,157 |
| `q36-35b-kapitsa` | 4,662 | 5,611 | 6,013 | 5,493 | 4,900 | 4,015 | 31,550 |
| `q38-27b-kapitsa` | 672 | 817 | 870 | 850 | 833 | 762 | 31,548 |
| `bolt-kapitsa` | 907 | 1,053 | 1,089 | 1,105 | 1,103 | 1,076 | 32,002 |

| Decode tok/s | 1K | 2K | 4K | 8K | 16K | 32K (max) | prompt tokens at 32K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `flash-kapitsa` | 89.5 | 87.0 | 82.4 | 80.5 | 79.8 | 78.2 | 31,550 |
| `flash-kapitsa-plain` | 82.7 | 78.6 | 78.9 | 79.4 | 77.6 | 77.6 | 31,550 |
| `lightning-kapitsa` | 206.2 | 197.6 | 197.1 | 177.0 | 152.5 | 117.0 | 32,006 |
| `lightning-kapitsa-chain` | 232.8 | 211.4 | 194.3 | 184.2 | 160.2 | 133.1 | 32,006 |
| `super-kapitsa` | 75.9 | 66.4 | 59.5 | 61.1 | 52.6 | 46.4 | 32,002 |
| `dory-kapitsa-perloop` | 109.1 | 105.9 | 101.1 | 94.8 | 88.7 | 75.5 | 32,006 |
| `dory-kapitsa` | 115.6 | 113.5 | 110.2 | 104.4 | 94.1 | 80.4 | 32,006 |
| `flint-kapitsa` | 256.0 | 250.6 | 241.5 | 229.4 | 210.5 | 179.6 | 29,157 |
| `q36-35b-kapitsa` | 268.4 | 262.1 | 257.3 | 239.7 | 218.3 | 174.9 | 31,550 |
| `q38-27b-kapitsa` | 96.5 | 84.0 | 88.4 | 77.1 | 70.9 | 61.0 | 31,548 |
| `bolt-kapitsa` | 114.0 | 108.3 | 108.3 | 111.3 | 102.1 | 79.5 | 32,002 |

## The setting sweeps

One run per setting, 4K / 16K prefill, decode T 0.6 and greedy.

| Model | Setting (label) | Prefill 4K / 16K | Decode | Greedy | |
| --- | --- | ---: | ---: | ---: | --- |
| Flash-Next | MTP depth 2, the default (`flash-kapitsa`) | 1,541 / 1,528 | 105.9 | 109.4 | **chosen** |
| | MTP depth 1 (`--spec 1`, `flash-kapitsa-mtp1`) | 1,443 / 1,428 | 86.0 | 91.4 | |
| | MTP depth 3 (`--spec 3`, `flash-kapitsa-mtp3`) | 1,684 / 1,674 | 102.0 | 107.0 | greedy loop |
| | MTP depth 4 (`--spec 4`, `flash-kapitsa-mtp4`) | 1,506 / 1,437 | 92.8 | 99.5 | decode loop |
| | `NI_KFLASH=0` (`flash-kapitsa-chain`) | 1,509 / 1,537 | 107.2 | 107.1 | inside the default's spread |
| Lightning | default (`lightning-kapitsa`) | 5,743 / 6,138 | 273.8 | 284.7 | **chosen** |
| | `NI_KFLASH=0` (`lightning-kapitsa-chain`) | 5,798 / 6,182 | 300.3 | 324.8 | faster, not an allowed switch: measured 3x below |
| Super-VL | default (`super-kapitsa`) | 1,092 / 1,168 | 87.1 | 87.6 | **chosen**; loop |
| | `NI_KFLASH=0` (`super-kapitsa-chain`) | 1,092 / 1,172 | 88.3 | 87.8 | inside the default's spread; loop |
| Dory | per-loop KV, 2 drafts per cycle, the default with it (`dory-kapitsa-perloop`) | 2,551 / 2,178 | 118.0 | 132.2 | **chosen** |
| | per-loop KV, `--spec 3` (`dory-kapitsa-perloop-spec3`) | 3,076 / 2,473 | 119.4 | 141.1 | the 07:49 re-run (below) |
| | per-loop KV, `--spec 4` (`dory-kapitsa-perloop-spec4`) | 2,553 / 2,147 | 108.6 | 138.0 | |
| | last-loop KV, Kapitsa's default (`dory-kapitsa`) | 2,973 / 2,452 | 120.0 | 120.2 | measured 3x: level at T 0.6, slower greedy |
| Flint | no speculation, the default (`flint-kapitsa`) | 5,716 / 5,184 | 263.3 | 263.2 | **chosen** |
| | prompt lookup, 4 drafts (`--spec 4`, `flint-kapitsa-ngram4`) | 5,788 / 5,037 | 203.6 | 198.0 | |
| 35B-A3B | DFlash2 drafter, the default (`q36-35b-kapitsa`) | 6,099 / 5,295 | 349.5 | 360.8 | **chosen** |
| | MTP head, depth 1 (`--dflash none`, `q36-35b-kapitsa-mtp`) | 5,109 / 4,498 | 256.5 | 248.9 | |
| | MTP head, depth 2 (`--dflash none --spec 2`, `q36-35b-kapitsa-mtp2`) | 4,974 / 4,413 | 219.2 | 218.2 | |
| 27B | DFlash2 drafter, 16-row block, the default (`q38-27b-kapitsa`) | 668 / 647 | 107.9 | 113.4 | **chosen**; slow window |
| | MTP head, depth 4 (`--dflash none`, `q38-27b-kapitsa-mtp`) | 700 / 676 | 78.1 | 74.0 | slow window |
| Bolt | default (`bolt-kapitsa`) | 1,070 / 1,064 | 149.9 | 159.0 | **chosen**; loop |
| | `NI_KFLASH=0` (`bolt-kapitsa-chain`) | 1,039 / 1,098 | 133.5 | 155.3 | |

The sweep labels are in [sweep-configs/](sweep-configs) (each extends its model's label in `bench/harness/configs/`),
and every run's file records its full command. `--spec` sets the MTP depth, Dory's drafts per cycle and Flint's prompt
lookup drafts (docs/usage.md).

**Lightning with chains.** `NI_KFLASH=0` (each cycle verifies a chain of the DSpark drafter's guesses instead of a KFlash
tree) decoded faster than the default in three interleaved runs each: 314.8 (278.4-316.5) against 287.3 (270.8-295.3)
tok/s at T 0.6, 315.4 against 301.6 greedy, and 133 against 117 tok/s at 32K context. Prefill is the same. The cell stays
on the default, since the switch is not a documented speed setting; making chains Lightning's default on the M5 Max
needs its own interleaved A/B and output check.

**Flash-Next's first block.** Before the context sweep was added (05:13-05:54, a 300 s quiet wait before each run),
three runs of `flash-kapitsa` read 1,524 / 1,517, 1,495 / 1,482 and 1,489 / 1,494 tok/s prefill and 104.5, 104.3 and
107.9 decode (median 1,495 / 1,494 and 104.5), and `flash-kapitsa-plain` 1,513 / 1,551 and 85.8. Those six runs are kept in
`results/`; the cells are the three runs of each label with the context sweep, as for every other model. Read on that
first block, Flash-Next's prefill gain is +28% / +31% and its decode -1.3%.

## Slow windows

Three stretches ran slow in every phase, prefill included, with no GPU contention recorded, no thermal or power limit,
and the same swap and memory pressure as the clean runs (each result's `thermal_start` and `vm_start`):

- 06:26-06:37, `dory-kapitsa-perloop-spec3-20261101-062546`: two requests stalled for about 250 s (a 512-token answer at
  2 tok/s), the 4K and 16K prompts read 3.5x and 2.2x slower. Re-run at 07:49 (`-074906`), the row in the sweep table.
- 06:49-06:59, `flash-kapitsa-20261101-064857` and `flash-kapitsa-plain-20261101-065428`: about 30% slower at every
  size of the sweep. Each was replaced by one more run (`-080723`, `-082721`).
- 07:56-08:02, the 27B's two sweep runs (`q38-27b-kapitsa-20261101-075632`, `q38-27b-kapitsa-mtp-20261101-075921`): prefill
  668 and 700 against 904 in the measured runs. Sweep runs only, both settings in the same window; the 27B's cell is the
  three runs at 08:58-09:24.

Each fell while the agent driving this session was working in the Claude desktop app; its renderer and WindowServer
were busy then, and WindowServer's GPU time is not counted by the harness's contention monitor. In the stretches where
it was idle, no run was slow. The cause is not established. The slow runs are kept in `results/` and are not cells.

## Contention and the quiet wait

The Claude desktop app's GPU process (`Claude Helper`), which hosts the agent driving these runs, held 5-9% of the GPU
while idle, so the harness's quiet-GPU wait never succeeded: its 300 s timeout expired before every run of the first two
blocks, and from 06:22 the runs used `--quiet-timeout 60` (each run records the wait and the busy process,
`provenance.gpu_quiet_wait_s` and `gpu_not_quiet`). The app was not declared ambient (`BENCH_GPU_AMBIENT` unset), so
its load counts as foreign and every cell is contention-flagged in some prefill or sweep phase (3.8-8.8% worst-phase mean
in prefill, 4.1-11.7% in the sweep), never in a decode phase (2.5% or less). The cells are published with those flags,
as the 2026-10-22 cells were (WebKit, 5.6-6.8% mean); the 2026-10-25 and 2026-10-26 cells were not flagged.

## Files

- `results/`: every run's result (66): the cells, the setting sweeps, the first Flash-Next block and the slow runs.
- `logs/`: each run's server log. `aborted/`: the server logs of two runs stopped by hand before any phase
  (06:19, to shorten the quiet wait; 06:23, started detached by mistake).
- `session/`: the runner's console output, block by block (`block1.log` the Flash-Next sweep, `block2.log` the first
  Flash-Next block and the Lightning and Super-VL sweeps, `block3.log` to `block6.log` the rest), with the load average at
  each run's start from block 5 on.
- `sweep-configs/`: the sweep labels, passed after the model's config.
- [machine.json](machine.json): the Mac, macOS, power settings, compiler and the binary's sha256.

Local paths are replaced by placeholders: `~` for the home folder, `<repo>` for the Kapitsa checkout, `<run-root>` for
the harness's output folder, `<results>` for this folder, `<host>` for the host name.
