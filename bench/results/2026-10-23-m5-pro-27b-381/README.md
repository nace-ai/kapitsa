# M5 Pro, 48 GB: the 27B after #371 and #381, 2026-10-23

[#371](https://github.com/nace-ai/kapitsa/pull/371) sends a suffix that holds the turn boundary and is under 25 rows
down the prefill path, and [#381](https://github.com/nace-ai/kapitsa/pull/381) moves the 27B's prefill attention to the
M5's tensor units (`qattn_t_g6d2s8`; its serve A/B: cold prefill 1.18x at 16K, 1.05x at 4K). This directory
re-measures the poster's M5 Pro Qwen3.8-27B cell on main after both merges, with the poster's protocol: the harness,
one config, prefill and decode from the same runs
(`compare_engines.py configs/kapitsa.json --only q38-27b-kapitsa --skip tools,agent`).

Only Kapitsa was re-run, as the poster's rules ask after a merge aimed at one engine. Every other engine's cell stays
its stored run (mlx-serve: the [2026-10-22 pair on `16f2b6a`](../2026-10-22-m5pro-48gb/README.md); Splash:
[2026-10-22](../2026-10-22-m5-pro-splash/README.md); the rest [2026-10-19](../2026-10-19-m5pro-48gb/README.md)).

- Mac: MacBook Pro M5 Pro 48 GB (20-core GPU), macOS 26.5.2 (25F84), AC power, High Power mode, 88-90% of memory free
  at each launch
- Build: main `fcd597b` (after #371 and #381), Release, Kapitsa 0.2.10, one binary for every launch (same sha256),
  `kapitsa serve` with the DFlash2 drafter (KFlash trees), `--ctx 32768 --slots 1`
- Model: qwen3.8-27b-4bit-mtp. Every server log shows the default 24/34 Neural Engine units and a cached Neural Engine
  program (ready in 0.1 s).
- Prompts: the harness corpus the stored cell used, 4,130-4,135 tokens at 4K and 16,285-16,288 at 16K (the stored cell:
  4,131-4,133 / 16,285-16,288), so the comparison is like for like
- Every launch in its own hold of the GPU lock. The runs are in `results/`, the server logs in `logs/`, `ps` snapshots
  at the start and end of every launch in `process-snapshots/` (home and checkout paths anonymized).

**The cell is the second session's three counted runs, and it is lower than the stored cell.** Session 1 (a discarded
first launch, then three counted) came in under the stored 649 / 81. Before anything else ran, the coordinator chose to
run the three counted launches once more with stricter gates (below), to take that set as the cell whatever it read,
and to run no third set. Session 2 reads the same decode as session 1 (76.57 against 76.59 sampled) and a little more
prefill (606 / 618 against 596 / 580). Both sessions are stored.

### Session 2, 12:15-18:16 (the cell)

Each launch held the GPU lock idle for 180 s first, and started only once no `kapitsa doctor`, `shasum`, `clang` or
foreign `kapitsa serve` was running. Launch 2 ran beside a video call in Microsoft Edge (57% mean foreign GPU, every
phase contention-flagged), so the coordinator had it repeated as launch 2b and the gate changed from "Edge and Windows
App closed" to "foreign GPU under 2% for 60 s" (Edge is often open with no call). Launches 3 and 2b waited about four
hours behind another team's GPU jobs, which had priority until 19:00.

| Run | Decode T=0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak footprint (MB) | Foreign GPU load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q38-27b-kapitsa-20261023-121500` (launch 1) | 80.16 (78.16-84.80) | 86.85 | 617 / 635 | 0.21 / 0.24 | 20,674 | 2.3% max, flagged: no |
| `q38-27b-kapitsa-20261023-125114` (launch 2, replaced) | 68.17 (53.24-83.13) | 73.04 | 473 / 504 | 0.21 / 0.24 | 20,674 | 70.2% max, flagged: contention |
| `q38-27b-kapitsa-20261023-171831` (launch 3) | 76.00 (63.81-77.26) | 83.87 | 604 / 618 | 0.22 / 0.24 | 20,675 | 5.2% max, flagged: no |
| `q38-27b-kapitsa-20261023-181308` (launch 2b) | 76.57 (71.50-78.92) | 83.68 | 606 / 591 | 0.22 / 0.23 | 20,676 | 2.5% max, flagged: no |
| **Median of 1, 3 and 2b** | **76.57** | **83.87** | **606 / 618** | 0.22 / 0.24 | 20,675 | |

Every decode phase has 5 of 5 valid answers. Launch 2 is kept and not counted. Launch 3 ran on the gate's timeout: the
GPU-quiet wait never saw foreign GPU under 2% for 60 s (the last reading was 6.4%, WebKit) and after three hours ran
anyway, as the protocol allows, so it is counted with that noted; the harness measured 2.1% mean and 5.2% max foreign GPU
in it and did not flag it. Launch 2b passed the GPU gate (under 2% for 60 s after a 236 s wait). Launches 3 and 2b found
the kernel cache gone and compiled the kernels again (2.96 s and 2.79 s); a 1 s sampler of the Metal compiler's CPU
(`session2-metal-compiler-cpu-*`) read at most 17.9% inside any measured phase.

Foreign CPU load in the start / end snapshots and in a 5 s sampler of processes above 30% CPU
(`session2-cpu-samples-*`):

| Launch | Foreign load |
| --- | --- |
| 1 (80.16) | nothing above 7% at start or end, nothing above 30% in the sampler (its file is empty, so not stored) |
| 2 (replaced) | Microsoft Edge helpers 30-67% and WindowServer 30-61% through the whole launch, `cameracaptured` 12-16% (the video call) |
| 3 (76.00) | the WebKit web view of an agent session's desktop app at 42-185% through the whole launch, that app's `node` host at 52-257% in four samples, a `modal app list` at 33% at start |
| 2b (76.57, the median) | the same web view at 54-200% through the whole launch, the `node` host at 70-153% in four samples; another checkout's `clang` builds (up to 99% each) for 10 s during warm-up, before the measured phases |

The two launches with the web view busy decode slower than launch 1 (76.0 and 76.6 against 80.2) and read the 4K prompt
slower (604 and 606 against 617). That is a correlation across three runs, not a measurement of the load's effect.

### Session 1, 11:38-11:59 (superseded)

| Run | Decode T=0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak footprint (MB) | Foreign GPU load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q38-27b-kapitsa-20261023-113849` (first launch, not counted) | 86.78 (67.08-92.01) | 90.56 | 614 / 630 | 0.21 / 0.23 | 20,674 | 4.3% max, flagged: no |
| `q38-27b-kapitsa-20261023-114312` | 81.61 (71.56-89.70) | 89.04 | 648 / 635 | 0.21 / 0.23 | 20,674 | 0.6% max, flagged: no |
| `q38-27b-kapitsa-20261023-115110` | 73.20 (68.33-76.30) | 80.42 | 561 / 548 | 0.20 / 0.23 | 20,672 | 2.1% max, flagged: no |
| `q38-27b-kapitsa-20261023-115529` | 76.59 (71.44-86.72) | 83.40 | 596 / 580 | 0.20 / 0.23 | 20,674 | 0.7% max, flagged: no |
| Median of the three counted | 76.59 | 83.40 | 596 / 580 | 0.20 / 0.23 | | |

No flags. The first launch of the fresh binary compiled the kernels (2.88 s) and is stored but not counted (chosen before
the runs). Launches 1-3 started only once no `kapitsa doctor`, `shasum` or `clang` was running, with no idle cool-down.
The snapshots show nothing above 7% CPU at any start except `launchd` at 25% before launch 2, and another checkout's
`clang` at 96% at the end of launch 2 (73.20, the slowest); no CPU sampler ran in this session.

### What changed on the poster

- **Decode: 81 to 77 tok/s** (81.37 to 76.57 sampled, 87.83 to 83.87 greedy). #371 changes which path a short cached
  suffix takes and #381 changes prefill attention, so neither is expected to move decode; both sessions read the same
  sampled median (76.6), and single runs span 73.2-81.6 in session 1 and 76.0-80.2 in session 2's counted runs.
- **Prefill: 649 to 606 tok/s at 4K (0.93x), 545 to 618 at 16K (1.13x).** #381 reported 1.05x at 4K and 1.18x at 16K
  from an interleaved A/B. These are separate sessions on different days and builds, not an A/B, so neither the 4K drop
  nor the 16K gain is measured as #381's.
- **Cached follow-up: 0.22 / 0.25 s to 0.22 / 0.24 s**, the path #371 changes. mlx-serve's stored pair answers sooner
  (0.14 / 0.18 s), and every counted run is slower than both, so the poster claims no follow-up win; the gain to cite
  stays cached suffix prefill 20-29% faster (probe, [#315](https://github.com/nace-ai/kapitsa/pull/315)).
- **mlx-serve now has the row's fastest decode.** Its stored pair reads 78.16 sampled and 86.23 greedy, 2.1% and 2.8%
  over this cell, so the poster's ≈ tie on this row ends and mlx-serve's 78 takes the marker. Kapitsa reads the 4K prompt
  1.31x as fast (606 against 464). The headline's count stays 32 of 36: the tie was not counted as a Kapitsa pair.
- Splash 1.4.0's stored cell (674 / 72; 618 at 16K) reads the 4K prompt 1.11x as fast, and at 16K the two are level
  (617.6 against 618.0). Two of the three counted runs read 16K under Splash's 618, so there is no 16K claim. Kapitsa
  decodes 6% faster (18% greedy).
