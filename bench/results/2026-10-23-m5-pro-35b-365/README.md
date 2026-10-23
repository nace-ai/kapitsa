# M5 Pro, 48 GB: the 35B-A3B after #365, 2026-10-23

[#365](https://github.com/nace-ai/kapitsa/pull/365) splits the head dim of the 35B-A3B's prefill attention over a pair
of simdgroups on the M5 (`qattn_t_g8d2s8`; its serve A/B: cold prefill 1.14x at 16K, 1.04x at 4K). This directory
re-measures the poster's M5 Pro Qwen3.6-35B-A3B cell on main after that merge, with the poster's protocol: the harness,
one config, prefill and decode from the same runs
(`compare_engines.py configs/kapitsa.json --only q36-35b-kapitsa --skip tools,agent`).

Only Kapitsa was re-run, as the poster's rules ask after a merge aimed at one engine. Every other engine's cell stays
its stored run ([2026-10-19](../2026-10-19-m5pro-48gb/README.md), [Splash, 2026-10-22](../2026-10-22-m5-pro-splash/README.md)).

- Mac: MacBook Pro M5 Pro 48 GB (20-core GPU), macOS 26.5.2 (25F84), AC power, display asleep, Windows App and Edge
  closed (checked before each launch), 82-85% of memory free at each launch
- Build: main `920542c` (#365's merge), Release, Kapitsa 0.2.10, `kapitsa serve` with the DFlash2 drafter (KFlash
  trees), `--ctx 32768 --slots 1`
- Model: qwen3.6-35b-a3b-4bit
- Prompts: the harness corpus the stored cell used, 4,132-4,135 tokens at 4K and 16,283-16,287 at 16K, so the
  comparison with the old cell (4,132 / 16,284-16,288) is like for like
- Two sessions of four launches each, every launch in its own hold of the GPU lock. In each session the first launch is
  stored but not counted (chosen before the runs), and the counted value is the median of the other three, each column
  its own median.

**The cell is the second session, which was not quieter than the first.** The first session's decode came in under the
old cell with another session's model conversion (`conv.py`, 5.9-9% CPU) in the process list, so the coordinator had the
whole protocol run again once that job ended, with every column taken from the new run. The `ps` snapshots stored with
the runs show the second session had more foreign CPU load, not less (below), so the choice of session rests on the
conversion alone, decided before session 2 ran, and is not a claim that session 2 was clean. Neither session has a GPU
contention flag; both are stored, and their prefill agrees (2,802 / 2,479 against 2,842 / 2,505). The decode half of the
cell should be read with the foreign load below in mind.

### Session 2, 08:36-08:49 (the cell)

| Run | Decode T=0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak footprint (MB) | Foreign GPU load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q36-35b-kapitsa-20261023-083654` (first launch, not counted) | 207.08 (192.52-236.84) | 219.94 | 2,822 / 2,442 | 0.05 / 0.06 | 22,266 | 0.7% max, flagged: no |
| `q36-35b-kapitsa-20261023-084020` | 196.96 (188.18-209.92) | 222.26 | 2,825 / 2,488 | 0.05 / 0.07 | 22,266 | 0% max, flagged: no |
| `q36-35b-kapitsa-20261023-084520` | 201.09 (197.30-203.29) | 216.95 | 2,802 / 2,479 | 0.05 / 0.06 | 22,257 | 0% max, flagged: no |
| `q36-35b-kapitsa-20261023-084805` | 222.64 (192.94-241.25) | 222.69 | 2,793 / 2,466 | 0.05 / 0.06 | 22,254 | 0% max, flagged: no |
| **Median of the three counted** | **201.09** | **222.26** | **2,802 / 2,479** | 0.05 / 0.06 | | |

Every decode phase has 5 of 5 valid answers, and no run has a contention or invalid flag. The binary was no longer
fresh (session 1 had compiled its kernels), but the first launch still compiled them again (2.96 s, tensor kernels
623 ms; the counted launches 7-13 ms). The Metal compiler's CPU never passed 10% in a measured phase.

Foreign CPU load in the snapshots (other sessions' jobs; the harness's contention check reads GPU only and saw 0-0.7%):

| Launch | At start | At end |
| --- | --- | --- |
| 0 (not counted) | `kapitsa doctor` (another checkout) 98.9%, `hf download` 6.1% | three `clang -cc1` 84-100%, `kapitsa doctor` 98.2% |
| 1 (196.96) | `kapitsa doctor` 96.1%, `hf download` 5.2% | `kapitsa doctor` 99.7%, `hf download` 4.1% |
| 2 (201.09, the median) | `shasum -a 256` on a GGUF 99.3%, `kapitsa doctor` 98.7% | `shasum` 99.4%, `kapitsa doctor` 97.0% |
| 3 (222.64) | `kapitsa doctor` 99.8% | the DSH web host (`node`) 209.9%, `kapitsa doctor` 98.1% |

So every counted launch started next to at least one process near 100% CPU. Whether `kapitsa doctor` touches the GPU
cannot be told from the snapshot. The decode spread (196.96-222.64, 13%) is not explained by it either way: the fastest
launch ends next to the busiest process. In session 1 the only process at or above 20% CPU in any snapshot is one
python job at 90.7% before launch 2.

### Session 1, 07:55-08:16 (superseded)

| Run | Decode T=0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak footprint (MB) | Foreign GPU load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q36-35b-kapitsa-20261023-075537` (first launch, not counted) | 200.02 (196.09-214.99) | 226.68 | 2,839 / 2,496 | 0.05 / 0.07 | 22,268 | 0% max, flagged: no |
| `q36-35b-kapitsa-20261023-080748` | 193.53 (186.02-223.76) | 214.50 | 2,842 / 2,507 | 0.05 / 0.07 | 22,268 | 0.6% max, flagged: no |
| `q36-35b-kapitsa-20261023-081138` | 195.07 (184.37-213.11) | 213.20 | 2,820 / 2,444 | 0.05 / 0.06 | 22,271 | 0% max, flagged: no |
| `q36-35b-kapitsa-20261023-081508` | 207.96 (180.46-250.67) | 221.10 | 2,845 / 2,505 | 0.06 / 0.07 | 22,259 | 0% max, flagged: no |
| Median of the three counted | 195.07 | 214.50 | 2,842 / 2,505 | 0.05 / 0.07 | | |

No flags either. Another session's model conversion (`conv.py convert --conns 8 --stream`, CPU and internal-SSD writes,
5.9-9% CPU in the snapshots) was in the process list at the start of launches 0, 1 and 2 and not launch 3, which decoded
fastest. The harness's contention check watches the GPU only, so it did not flag this. The first three launches
compiled the kernels (2.8-2.9 s, tensor kernels 592-621 ms); the compiler's CPU never passed 2% in a measured phase.

The runs are in `results/`, the server logs in `logs/`. The harness's results do not record other processes, so the
foreign jobs above are evidenced by `process-snapshots/`: the ten busiest processes (`ps`, by CPU) at the start of every
launch, and in session 2 also at its end (home paths anonymized). `conv.py` is in session 1's launches 0-2 and not 3;
`hf download` is in session 2's launches 0 and 1 only; `kapitsa doctor` is in every session-2 snapshot and no session-1
snapshot.

### What changed on the poster

- **Prefill: 2,729 to 2,802 tok/s at 4K (1.03x), 2,194 to 2,479 at 16K (1.13x)**, against the median of three runs on
  `e6e1f46` after #362; #365 reported 1.04x and 1.14x-1.16x with its own serve A/B.
- **Decode: 214 to 201 tok/s** (214.4 to 201.1 sampled; greedy 222.7 to 222.3, unchanged). #365 changes prefill
  attention only, and its math500 legs read the same median decode on both sides (263.7 vs 262.3 tok/s). The sampled
  medians of single runs on this Mac span 193.5-222.6 across both sessions and 209.8-219.3 on `e6e1f46`, so the sampled
  drop is inside run-to-run spread and not taken to be #365's.
- Splash 1.4.0's stored cell (3,514 / 181) still reads the 4K prompt faster (1.25x); Kapitsa decodes 11% faster.
