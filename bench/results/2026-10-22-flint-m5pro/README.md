# M5 Pro, 48 GB: Recursive Flint 40B-A2.5B on Kapitsa and rec, 2026-10-22

The M4 Pro's Flint measurement ([`2026-10-22-flint-m4pro`](../2026-10-22-flint-m4pro/README.md)), repeated on the M5
Pro with the same protocol:

- Kapitsa `6c13588` (main: v0.2.7 plus #304, a benchmark-only commit) on Recursive's 4-bit MLX export
  ([RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4) @ 3b93f905).
- rec 0.1.0, Recursive's own engine ([recursive-oss/rec](https://github.com/recursive-oss/rec), MLX 0.32), on the
  same files.

llama.cpp b11804 was not run here: its Q4_1 GGUF (24.2 GiB) did not fit next to the MLX files within this Mac's free
disk floor (see Conditions).

There are two measurements. **The poster's protocol**
(`bench/harness/compare_engines.py configs/flint.json --skip agent`) ran Kapitsa and rec in two passes in opposite
order (Kapitsa then rec, rec then Kapitsa), all four runs in one hold of the GPU lock, so a pass-order effect cancels
in the means. **A context sweep** then measured both engines from 1K to 128K tokens of prompt.

As on the M4 Pro, every run uses `bench.py --always-reasons` (Flint always reasons first), and decode samples at the
model card's temperature 1.0 (top_p 0.95, top_k 20).

## The poster's protocol

| Pass | Engine | Run | Decode T=1.0 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Follow-up TTFT 4K / 16K (s) | Peak (GB) | Foreign GPU load |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (fwd) | Kapitsa | `flint-kapitsa-20261022-161319` | 148.68 (145.1-150.2) | 151.94 | 1,392 / 1,125 | 0.12 / 0.15 | 23.25 | 3.56% mean, 16.3% max, flagged |
| 1 (fwd) | rec 0.1.0 | `flint-rec-20261022-161536` | 144.70 (137.1-145.4) | 152.07 | 2,130 / 1,926 | 0.24 / 0.31 | 29.31 | 2.99% mean, 12.9% max, flagged |
| 2 (rev) | rec 0.1.0 | `flint-rec-20261022-161819` | 145.11 (144.4-145.9) | 151.64 | 2,145 / 1,955 | 0.32 / 0.24 | 28.83 | 4.98% mean, 28.9% max, flagged |
| 2 (rev) | Kapitsa | `flint-kapitsa-20261022-162029` | 142.80 (137.8-144.0) | 145.67 | 1,356 / 1,091 | 0.11 / 0.14 | 23.25 | 10.15% mean, 27.1% max, flagged |
| | **Kapitsa mean** | | **145.74** | **148.81** | **1,374 / 1,108** | **0.12 / 0.14** | **23.25** | |
| | **rec 0.1.0 mean** | | **144.91** | **151.86** | **2,138 / 1,941** | **0.28 / 0.28** | **29.07** | |

The runs are in `results/`, their server logs in `logs/`. Decode is the median of 5 runs of 512 tokens (the range in
brackets), prefill the harness's 4K and 16K prompts (3,872-3,875 and 15,066-15,068 of Flint's tokens), follow-up TTFT
the median time to the first token of a cached second turn. Local paths in them are replaced by placeholders: `~` for
the home folder, `<repo>` for the Kapitsa checkout, `<run-root>` for the run directory, `<scratch>` for this session's
folder of engine binaries and scripts.

- **Prefill.** rec reads prompts 1.56x as fast as Kapitsa at 4K (2,138 against 1,374 tok/s) and 1.75x at 16K (1,941
  against 1,108). This is the reverse of the M4 Pro, where Kapitsa read 1.22x and 1.17x as fast as rec.
- **Decode.** A tie: Kapitsa 145.7 against rec 144.9 tok/s at T 1.0 (0.6% apart, inside both engines' spread), and
  greedy 148.8 against 151.9 (rec 2% ahead). The contention leans against Kapitsa here: its decode phases are flagged
  in both runs (5.4% and 14.9% mean foreign load at T 1.0, 7.4% greedy in the second), rec's in neither, and Kapitsa's
  second run, the most contended, is its slowest (142.8).
- **Follow-up turns.** Kapitsa starts a cached second turn 2.4x sooner than rec at 4K (0.12 against 0.28 s) and 2.0x
  sooner at 16K.
- **Memory.** Kapitsa peaks at 23.3 GB, rec at 29.1.
- **Against the M4 Pro.** Kapitsa decodes 1.14x and reads a 4K prompt 1.42x as fast as there (127.9 and 968 tok/s);
  rec decodes 1.15x and reads 2.69x as fast (126.6 and 796). Why rec's prefill gains so much more on the M5 Pro was not
  measured.

**The poster's Flint cells are the means**, as prefill at 4K / decode at T 1.0: Kapitsa 1,374 / 146, rec 2,138 / 145.
The poster ranks a row by its decode value, so by that rule the row counts as Kapitsa's, but the margin is inside the
runs' spread and greedy goes the other way: read it as a tie on decode, with rec 1.56x ahead on prefill.

rec's second run (`flint-rec-20261022-161819`) has one more flag: its third cached 16K follow-up stopped at the
length limit with no content, so the harness stopped that series. Its 16K follow-up TTFT is the median of the two
valid samples (0.237 s).

## Context sweep: 1K to 128K tokens

| Prefill (tok/s) | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kapitsa | 1,151 | 1,382 | 1,302 | 1,288 | 1,102 | 847 | 578 | 350 |
| **rec 0.1.0** | **1,582** | **2,144** | **2,112** | **2,108** | **1,934** | **1,657**¹ | **1,250**¹ | **817** |
| <sub>Kapitsa ÷ rec</sub> | <sub>0.73×</sub> | <sub>0.64×</sub> | <sub>0.62×</sub> | <sub>0.61×</sub> | <sub>0.57×</sub> | <sub>0.51×</sub> | <sub>0.46×</sub> | <sub>0.43×</sub> |

| Decode (tok/s) | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Kapitsa** | **144.8** | **138.3** | 130.4 | **129.4** | **118.0** | **97.4** | **74.3** | **49.5** |
| rec 0.1.0 | 143.1 | 137.1 | **132.5** | 122.7 | 107.5¹ | 88.6¹ | 66.4 | 43.1 |
| <sub>Kapitsa ÷ rec</sub> | <sub>1.01×</sub> | <sub>1.01×</sub> | <sub>0.98×</sub> | <sub>1.05×</sub> | <sub>1.10×</sub> | <sub>1.10×</sub> | <sub>1.12×</sub> | <sub>1.15×</sub> |

¹ From the rerun; both runs' values are in the table under *Contention and reruns*.

- **Prefill.** rec reads faster at every length, and its lead grows with the prompt: 1.4x at 1K, 1.6x at 4K, 1.75x at
  16K, 2.3x at 128K. At 128K, rec keeps 39% of its 4K speed and Kapitsa 27%. On the M4 Pro, Kapitsa led at every
  length (1.09-1.22x) and kept 31% at 128K.
- **Decode.** The M4 Pro's shape again, but closer: level up to 8K (0.98-1.05x), then Kapitsa 1.10-1.15x ahead from
  16K. That lead is the size of the noise: Kapitsa's own 16K decode differs by 7% between its two sweeps, and the
  long-context ratios read as about ±10% (below). At 128K, Kapitsa keeps 38% of its 4K decode speed, rec 33%.
- **Against the M4 Pro** at 128K: Kapitsa reads 1.17x as fast (350 against 300 tok/s) and decodes 1.08x (49.5 against
  45.7); rec reads 3.1x as fast (817 against 262) and decodes 1.22x (43.1 against 35.2).

The sweep's 2K and 4K prefill points bracket the harness's 4K numbers above (Kapitsa 1,382 and 1,302 against 1,374;
rec 2,144 and 2,112 against 2,138).

How it ran (`context-sweep/`): `ctx_sweep.py` against each engine's server, started by `sweep_engine.sh` with a long
context (Kapitsa `--ctx 262144` with `NI_DISK_CACHE=0`, rec `--context 262144`), one engine at a time, each in its own
hold of the GPU lock after `run_sweep.sh` waited for 40% of memory to be free, and each warmed first by one 2K
request. The prompt, prefill and decode are defined as on the M4 Pro: one cold request per size, a coding agent's
context (`bench/corpus.txt`, then the repository's source files in path order, cut to the size in Flint's tokens) with
a random nonce at its start, then a 256-token reply at T 1.0; prefill is the prompt's tokens over the time to the first
streamed token, decode the stream's rate between its 16th and its 4th-from-last event.

Unlike the M4 Pro's sweep, the prompt source was frozen before the sweeps (`git archive` of `6c13588`), so both engines
saw the same 1,645,932 tokens. Kapitsa's first attempt stopped before its first request, because the archive kept
`bench/corpus.txt` (a symlink) without its target (`context-sweep/logs/kapitsa-first-attempt-missing-corpus.log`); the
target was added from the same commit, and the sweep ran after rec's.

**Contention and reruns.** This sweep has a contention sampler: `gpu_hogs.py monitor` logged non-engine GPU load every
2 s (`context-sweep/foreign-gpu/`). Its only source was Microsoft Edge, and it was heavy in places: up to 85% for a
few seconds of Kapitsa's 128K prefill, and 53% on average through rec's 32K prefill. The points with load in their
windows were run once more (Kapitsa 2K-16K, rec 16K-64K, the `-rerun` logs), and each table cell takes the run with
the lower mean foreign load in that metric's window (the first sweep on a tie). `context-sweep/choose_cells.py`
rebuilds that choice from the committed sampler files and logs (`python3 choose_cells.py` prints it, `--write` stores
it as `sweep.json`'s `chosen`, with both runs' values and load for every cell). rec's windows come from its log's
request timestamps; Kapitsa's log has none, so its windows are laid back from the sweep's end with its logged
durations, about 2 s of error.

The prefill windows from 16K up hold 5-180 samples, but a 256-token decode lasts about 2 s and holds one or two of
the sampler's 2 s readings, so **the decode choice rests on one or two samples and is close to arbitrary**. Both runs
of every rerun point:

| Point | First sweep (foreign load) | Rerun (foreign load) | Rerun vs first | Table uses |
| --- | ---: | ---: | ---: | --- |
| Kapitsa 2K prefill / decode | 1,382 (0%) / 138.3 (11%) | 1,333 (0%) / 133.9 (22%) | -3.5% / -3.2% | first / first |
| Kapitsa 4K prefill / decode | 1,302 (24%) / 130.4 (16%) | 1,334 (28%) / 130.5 (43%) | +2.5% / +0.1% | first / first |
| Kapitsa 8K prefill / decode | 1,288 (7%) / 129.4 (3%) | 1,227 (39%) / 123.6 (57%) | -4.7% / -4.5% | first / first |
| Kapitsa 16K prefill / decode | 1,102 (11%) / 118.0 (8%) | 1,055 (46%) / 109.3 (47%) | -4.3% / -7.4% | first / first |
| rec 16K prefill / decode | 1,934 (10%) / 100.6 (23%) | 1,897 (22%) / 107.5 (13%) | -1.9% / +6.9% | first / rerun |
| rec 32K prefill / decode | 1,466 (53%) / 78.7 (38%) | 1,657 (13%) / 88.6 (10%) | +13.0% / +12.6% | rerun / rerun |
| rec 64K prefill / decode | 1,084 (37%) / 66.4 (5%) | 1,250 (16%) / 64.5 (12%) | +15.3% / -2.9% | rerun / first |

Kapitsa's reruns were more contended than its first sweep, so all its cells are from the first sweep. The rerun raised
rec's 32K and 64K prefill by 13% and 15% and its 16K and 32K decode by 7% and 13%, the cells where the first sweep's
load was 23-53%; the other rerun points moved 2-5%, and Kapitsa's 16K decode 7%.

Each sweep cell is a single request: about ±3% on a quiet GPU (as on the M4 Pro), more where the load was. Above 8K,
most chosen prefill windows still had some (up to 16% mean), so the long-context ratios between the engines are good
to about ±10%. Each point's runs, warm-ups, chosen run and both runs' foreign load are in `context-sweep/sweep.json`,
the servers' logs in `context-sweep/logs/`. 256K was not run.

## Charts

Three 16:9 charts in the README poster's dark style, each at 1920x1080 and, as `-2x`, 3840x2160, drawn from the
tables above and `context-sweep/sweep.json` by `charts/flint_charts.py` (`python3 flint_charts.py`, needs Chrome):
[headline](charts/flint-kapitsa-m5pro-headline.png),
[decode by context](charts/flint-kapitsa-m5pro-decode-by-context.png),
[prefill by context](charts/flint-kapitsa-m5pro-prefill-by-context.png). They show Kapitsa and rec only:
llama.cpp was not run on this Mac.

## Conditions

MacBook Pro (16-inch, 2026), `Mac17,8`, M5 Pro (20-core GPU), 48 GB, macOS 26.5.2 (25F84), on AC power in High Power
mode, with 6-7.5 GB of swap in use at the starts. Each engine load waited for the GPU lock `/tmp/kapitsa-gpu.lock` and
for 40% of memory to be free.

**Contention.** Microsoft Edge used the GPU during every run: 3-10% on average in the poster runs, by the harness's own
sampling, so all four are flagged (`CONTENTION: sustained foreign GPU load > 5%` in at least one phase). Kapitsa's
second run had the most (10.15% mean, in every phase). Other workers' benchmarks held the GPU lock between this
session's passes, never during them.

**Kapitsa** ran one binary in every run and both sweeps (sha256 `7781ef566e01…`), built Release from `6c13588`, its
source clean, with `--ctx 32768 --slots 1` and plain decoding in the poster runs. Of main's later commits up to this PR, one changes
the engine: [#303](https://github.com/nace-ai/kapitsa/pull/303) runs Flint's decode merges in separate dispatches on
macOS 14, which lacks Metal 3.2's device-scope fences. On macOS 15 and newer (this Mac runs 26.5.2) it keeps the
fused path measured here unless `NI_FLINT_SPLIT=1` is set. The other commits are docs, results and the poster.

**rec 0.1.0** ran as its release binary (`rec-darwin-arm64`, sha256 `22f08e8505e5…`, matching the release's `.meta`),
`rec serve --context 32768` in the poster runs, on APFS clones of the same int4 files.

**Not run:** llama.cpp. Its Q4_1 GGUF is 24.2 GiB, and this Mac's data volume had 24-27 GiB free after the MLX files,
under the 30 GiB floor this session kept. Also not run: Ollama, LM Studio and the other MLX servers (none has Flint
here), and 256K context.
