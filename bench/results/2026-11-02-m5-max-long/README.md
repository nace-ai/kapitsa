# Results, 2026-11-02: MacBook Pro M5 Max, 128 GB (Kapitsa from 1K to 512K context)

Kapitsa's prefill and decode from 1K tokens of context to each model's limit: 128K on Clover, 1M on Lightning and
Super-VL (measured to 512K), 256K on the rest. Every model at its default, the labels of its
[2026-11-01 cells](../2026-11-01-m5-max-kapitsa/README.md), with the server's context set to that limit (`--ctx`).
Kapitsa 0.2.26: the Release build of `fd6f977`, which changes only `bench/harness` and `tests/` over main `b7a9711`.
Written by [bench/harness](../../harness/README.md), one run at a time under the GPU lock:
`compare_engines.py configs/<model>.json sweep-configs/long.json --only <label> --agentic --skip
greedy,prefill,tools,agent,cache,sustain --quiet-timeout 60`. MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU,
128 GB, macOS 26.4 (25E246), every run used here on AC power in High Power mode under `caffeinate -is`
([machine.json](machine.json), [session/power.log](session/power.log)).

**What a point is.** The harness's context sweep: at each size, two cold requests, each a document of that length cut
from the long corpus (`corpus/make_corpus.py --long`, sha256 `4339c1d0…`; its first 650,000 characters are the frozen
corpus) followed by the decode phase's 512-token coding question (`--sweep-task code`) at T 0.6 (Flint at its card's
T 1.0). Prefill is the prompt's tokens over the time to its first token; decode is the middle of the answer. A point is
the median of every request at that size: two from one run, four where a second run was made (below). The sizes are
approximate: each model's tokenizer reads the same text in its own number of tokens, so 128K is 122-133K tokens, 256K
is 240-271K and 512K is 512K (the prompt-token column).

Prefill, tok/s:

| Prefill tok/s | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B | 4,629 | 5,630 | 5,613 | 5,356 | 4,255 | 3,418 | 2,667 | 1,756 | 995 | - |
| Clover 35B-A3B | 4,537 | 5,328 | 5,773 | 5,430 | 4,769 | 3,853 | 2,784 | 1,828 | - | - |
| Qwen3.8 27B | 755 | 848 | 880 | 871 | 854 | 778 | 658 | 492 | 286 | - |
| Flash-Next 180B | 1,511 | 1,477 | 1,631 | 1,539 | 1,476 | 1,417 | 1,326 | 1,242 | 1,087 | - |
| Lightning 30B | 5,138 | 5,680 | 5,411 | 5,370 | 5,165 | 4,645 | 3,597 | 2,488 | 1,482 | 823 |
| Super-VL 120B | 956 | 986 | 1,176 | 1,188 | 1,178 | 1,141 | 1,050 | 859 | 580 | 337 |
| Bolt 74B | 916 | 1,056 | 1,090 | 1,089 | 1,081 | 1,040 | 965 | 792 | 549 | - |
| Dory1.0 4B | 2,448 | 2,295 | 2,286 | 2,226 | 2,061 | 1,557 | 1,156 | 731 | 405 | - |
| Flint 40B | 4,266 | 5,650 | 5,868 | 5,315 | 4,891 | 3,760 | 2,956 | 2,049 | 1,241 | - |

Decode, tok/s:

| Decode tok/s | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B | 330 | 327 | 316 | 309 | 255 | 245 | 174 | 128 | 80.2 | - |
| Clover 35B-A3B | 306 | 325 | 350 | 254 | 244 | 224 | 176 | 130 | - | - |
| Qwen3.8 27B | 77.6 | 101 | 93.5 | 69.3 | 83.6 | 74.1 | 49.5 | 34.2 | 27.6 | - |
| Flash-Next 180B | 97.3 | 94.9 | 92.8 | 92.1 | 92.5 | 91.5 | 90.2 | 85.4 | 75.8 | - |
| Lightning 30B | 240 | 240 | 231 | 213 | 182 | 146 | 103 | 61.1 | 37.6 | 21.0 |
| Super-VL 120B | 97.4 | 68.3 | 73.5 | 67.2 | 63.3 | 53.0 | 47.0 | 38.6 | 24.1 | 16.0 |
| Bolt 74B | 142 | 147 | 145 | 117 | 112 | 98.5 | 78.3 | 47.5 | 31.1 | - |
| Dory1.0 4B | 111 | 108 | 105 | 97.1 | 84.7 | 70.0 | 51.6 | 35.2 | 20.5 | - |
| Flint 40B | 258 | 252 | 246 | 233 | 212 | 181 | 141 | 98.5 | 62.4 | - |

Prompt tokens (the median request's):

| Prompt tokens | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B | 1.1K | 2.1K | 4.2K | 8.3K | 16K | 33K | 65K | 131K | 259K | - |
| Clover 35B-A3B | 1.1K | 2.1K | 4.2K | 8.3K | 16K | 33K | 65K | 127K | - | - |
| Qwen3.8 27B | 1.1K | 2.1K | 4.2K | 8.3K | 16K | 33K | 65K | 131K | 259K | - |
| Flash-Next 180B | 1.1K | 2.1K | 4.2K | 8.3K | 16K | 33K | 65K | 131K | 259K | - |
| Lightning 30B | 1.1K | 2.2K | 4.2K | 8.4K | 17K | 33K | 66K | 133K | 271K | 512K |
| Super-VL 120B | 1.1K | 2.2K | 4.2K | 8.4K | 17K | 33K | 66K | 133K | 271K | 512K |
| Bolt 74B | 1.1K | 2.2K | 4.2K | 8.4K | 17K | 33K | 66K | 133K | 255K | - |
| Dory1.0 4B | 1.1K | 2.2K | 4.2K | 8.4K | 17K | 33K | 66K | 133K | 255K | - |
| Flint 40B | 1.0K | 2.0K | 3.9K | 7.7K | 15K | 30K | 61K | 122K | 240K | - |

Requests per point: Qwen3.6 35B-A3B 4 to 128K; Clover 35B-A3B 2; Qwen3.8 27B 2; Flash-Next 180B 4 to 128K; Lightning 30B 4 to 128K; Super-VL 120B 2; Bolt 74B 2; Dory1.0 4B 4 to 128K; Flint 40B 2.

## The decode here is not the decode cell

The answer is the decode cell's (a 512-token coding answer at T 0.6), but here it comes after a document. With a document
in front of it a drafter guesses fewer tokens right, so a drafted model's 1K point can sit well below its cell: the 27B's
DFlash2 drafter got 3.50 and 3.87 tokens per verify cycle at 1K, against 5.6-6.7 after the decode phase's 85-token
prompt in the same run, and the 27B decoded 78 tok/s at 1K against its 143.3 cell. Flint, which drafts nothing, reads
258 at 1K against its 264.9 cell. These are the numbers [README.md](../../../README.md)'s context charts plot, and the
charts' footers say so.

## Where the long-context time goes

The server logs give each answer's decode time and its verify cycles. Per cycle (decode ms over cycles; Flint, with no
speculation, one token per step), against one read of the model's whole KV cache at that length (the server's KV bytes
per token times the prompt, at an assumed 550 GB/s):

| Model | KV per token | ms per step at ~1K | at ~32K | at ~128K | at 256K-512K | one KV read there | growth over one KV read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B | 22 KB | 11 | 16 | 30 | 47 (259K) | 10.6 ms | 3.5x |
| Clover 35B-A3B | 20 KB | 11 | 16 | 29 | 29 (127K) | 4.7 ms | 3.8x |
| Qwen3.8 27B | 68 KB | 48 | 68 | 123 | 219 (259K) | 32.8 ms | 5.2x |
| Flash-Next 180B | 31 KB | 26 | 27 | 29 | 31 (259K) | 14.9 ms | 0.3x |
| Lightning 30B | 6 KB | 14 | 24 | 53 | 161 (512K) | 5.7 ms | 25.7x |
| Super-VL 120B | 8 KB | 48 | 67 | 109 | 252 (512K) | 7.6 ms | 26.7x |
| Bolt 74B | 8 KB | 29 | 45 | 84 | 134 (255K) | 3.8 ms | 27.9x |
| Dory1.0 4B | 56 KB | 23 | 35 | 67 | 120 (255K) | 26.6 ms | 3.6x |
| Flint 40B | 28 KB | 4 | 6 | 10 | 16 (240K) | 12.5 ms | 1.0x |

Flint's step grows by about one KV read, so its single-row decode attention runs near the memory's speed. The verify
cycles, which run a draft tree of about 16 rows, grow 3.5-5.2x more than one KV read on the Qwen models and Dory and
26-28x more on the Nemotron models, whose KV is small (6-8 KB per token); Flash-Next's MTP head verifies a short chain,
and its step grows by less than one KV read at the server's figure. The verify paths are where long-context decode can
gain most; nothing is changed here.

## Second runs

The first runs of the 35B, Lightning, Flash-Next and Dory read 5-30% below the
[2026-11-01 sweep](../2026-11-01-m5-max-kapitsa/README.md#from-1k-to-32k-context) in prefill at 1K-32K (Dory at 2K: 2,125
against 3,143). Two things differ from that sweep: the server's context (`--ctx 256K` or `512K` against 32K) and the
question after the document. Five interleaved Dory runs at 2K / 4K / 16K (15:45-15:55, `ab/`) separate them:

| Dory, prefill tok/s at 2K / 4K / 16K | Run 1 | Run 2 |
| --- | ---: | ---: |
| `--ctx 262144`, the coding question (`dory-ab-ctx256k-code`) | 3,194 / 3,174 / 2,253 | 3,193 / 3,181 / 2,376 |
| `--ctx 32768`, the coding question (`dory-ab-ctx32k-code`) | 3,200 / 3,096 / 2,419 | 2,774 / 2,579 / 2,212 |
| `--ctx 32768`, the 2026-11-01 summary question (`dory-ab-ctx32k-summary`) | 2,486 / 2,110 / 2,089 | |

The large context costs nothing, and the long sweep's Dory run (14:09, 2,125 / 2,213 / 1,955) read slower than the
same label 90 minutes later. The cause is not established: no run recorded a thermal or power limit, that run had the
battery at 5-7% on a 96 W charger (Flint, at 11% on the same charger at 12:47, read as on 2026-11-01), and two identical
`--ctx 32768` runs here differ by up to 17%. So the four models got a second run of their 1K-128K sizes on the 140 W
charger (15:56 on, `-r2` labels in [sweep-configs/rerun.json](sweep-configs/rerun.json)), and their points to 128K are the
median of both runs' four requests. Each run's own points are in its result file. Against the first runs, the second read in prefill
-5% to +2% on the 35B, +4% to +7% on Flash-Next (1K-32K), +8% to +12% on Lightning (1K-32K) and +8% to +21% on Dory, so
the first runs of the last three were slow and the medians sit between. Dory's second run (16:17) ran a build of the
same commit with an uncommitted change to Nemotron-H's verify attention (`src/kernels/tensor/nh_attn_t.metal`, a later
pull request), which Dory's model does not run (it compiles none of the tensor kernels).

## Runs

| Run | Local time | What it gives | Flags |
| --- | --- | --- | --- |
| [`q36-35b-kapitsa-256k-20261102-055311`](results/q36-35b-kapitsa-256k-20261102-055311.json) | 05:53-06:07 | 1K-256K | loop, contention |
| [`flint-kapitsa-256k-20261102-060729`](results/flint-kapitsa-256k-20261102-060729.json) | 06:08-06:20 | not used: T 0.6, reasoning returned | contention, invalid |
| [`clover-kapitsa-128k-20261102-062007`](results/clover-kapitsa-128k-20261102-062007.json) | 06:21-06:25 | 1K-128K | loop, contention |
| [`lightning-kapitsa-512k-20261102-062543`](results/lightning-kapitsa-512k-20261102-062543.json) | 06:26-06:36 | 1K-256K (its 512K overflowed) | contention, overflow |
| [`bolt-kapitsa-256k-20261102-063655`](results/bolt-kapitsa-256k-20261102-063655.json) | 06:38-06:49 | 1K-128K (its 256K overflowed) | loop, stopped early, contention, overflow |
| [`flash-kapitsa-256k-20261102-064953`](results/flash-kapitsa-256k-20261102-064953.json) | 06:50-07:07 | 1K-256K | loop, stopped early |
| [`super-kapitsa-512k-20261102-070742`](results/super-kapitsa-512k-20261102-070742.json) | 07:08-07:36 | 1K-256K but 2K (a loop; its 512K overflowed) | loop, stopped early, overflow |
| [`q38-27b-kapitsa-256k-20261102-073636`](results/q38-27b-kapitsa-256k-20261102-073636.json) | 07:37-11:10 | 1K-128K (its 259K crossed a sleep) | stopped early, contention |
| [`dory-kapitsa-perloop-256k-20261102-111035`](results/dory-kapitsa-perloop-256k-20261102-111035.json) | 11:11-11:42 | not used: on battery | loop, battery |
| [`flint-kapitsa-256k-20261102-124710`](results/flint-kapitsa-256k-20261102-124710.json) | 12:47-12:58 | 1K-256K |  |
| [`bolt-kapitsa-256k-top-20261102-125828`](results/bolt-kapitsa-256k-top-20261102-125828.json) | 12:59-13:15 | 256K |  |
| [`lightning-kapitsa-512k-top-20261102-131529`](results/lightning-kapitsa-512k-top-20261102-131529.json) | 13:15-13:37 | 512K |  |
| [`q38-27b-kapitsa-256k-top-20261102-133745`](results/q38-27b-kapitsa-256k-top-20261102-133745.json) | 13:38-14:09 | 256K | stopped early |
| [`dory-kapitsa-perloop-256k-20261102-140933`](results/dory-kapitsa-perloop-256k-20261102-140933.json) | 14:09-14:43 | 1K-256K | loop, contention |
| [`super-kapitsa-512k-top-20261102-144400`](results/super-kapitsa-512k-top-20261102-144400.json) | 14:44-15:37 | 2K, 512K | loop, contention |
| [`q36-35b-kapitsa-256k-r2-20261102-155623`](results/q36-35b-kapitsa-256k-r2-20261102-155623.json) | 15:57-16:02 | 1K-128K, second run | loop, contention |
| [`lightning-kapitsa-512k-r2-20261102-160217`](results/lightning-kapitsa-512k-r2-20261102-160217.json) | 16:03-16:07 | 1K-128K, second run | contention |
| [`flash-kapitsa-256k-r2-20261102-160726`](results/flash-kapitsa-256k-r2-20261102-160726.json) | 16:08-16:17 | 1K-128K, second run | stopped early, contention |
| [`dory-kapitsa-perloop-256k-r2-20261102-161718`](results/dory-kapitsa-perloop-256k-r2-20261102-161718.json) | 16:17-16:28 | 1K-128K, second run | loop, contention |

Not used: `flint-kapitsa-256k-20261102-060729` ran at T 0.6 without `--always-reasons` (the long label's arguments
replaced its parent's), so its decode phase is INVALID (reasoning returned). `dory-kapitsa-perloop-256k-20261102-111035`
ran on battery (the charger came out at 10:28; 74% to 14%). `q38-27b-kapitsa-256k-20261102-073636`'s 259K point crossed an
idle sleep at 07:54 (the Mac's `sleep` is 1 minute and nothing held it awake then): its first request took 9,723 s by
the server's clock and 1,087 s by the harness's, whose monotonic clock stops in sleep. Its points to 128K are used: the
second 128K request ended about 10 s before the sleep, and the server's times for both 128K requests (263.8 s, 265.7 s)
match the harness's (264.8 s, 267.7 s). The `max` size overflowed the Nemotron models' context (Bolt 264,365 tokens
against 262,142, Lightning 532,608 and Super-VL 532,602 against 524,286: their tokenizer reads the long corpus' later
files at about 1.07 tokens per estimated one), so their top points are `-top` runs at 240,000 and 480,000 estimated tokens.
Super-VL's first 2K point (974 and 677 tok/s) was a repetition loop; its `-top` run re-measured it.

## Flags

Repetition loops (distinct 4-grams below 0.5; recorded per sweep answer from 12:47 on, after the first runs): Dory's 4K and
64K points each have one looping answer, which decoded at its pair's speed (95 and 97, 51 and 51 tok/s:
Dory's self-drafts gain nothing from a loop); Super-VL's re-run 2K has one (66 and 71 tok/s, against 974 for the loop it
replaced). Super-VL's 1K point (median 97) is 120 and 74 tok/s; the 120 answer verified 6.55 tokens per cycle against 3.19,
which looks like a partial loop that the first runs did not yet check for. Answers that stopped before 512 tokens
(`finish_reason=stop`, decode measured on the shorter answer): Bolt 1K, Flash-Next 64K, Super-VL 4K, the 27B 4K, 16K,
32K and one of its 259K answers (175 tokens; 27.4 against the other's 27.1 tok/s). Contention: the Claude desktop app
driving the session (`Claude Helper`) held 1-10% of the GPU, so most sweeps are contention-flagged (worst: Clover's,
9.6% mean); the harness's quiet wait timed out at 60 s before most runs.

## Files

- `results/`: every run (the long sweep, the `-top` and `-r2` runs, and the runs not used, above). `ab/`: the Dory
  context A/B's results.
- `logs/`: each run's server log, with its verify cycles per answer.
- `session/`: the runner's console output (`block1.log` 05:53-11:42, `block2.log` 12:47-15:37, `ab.log` 15:45-15:55,
  `block3.log` 15:56-16:29, the second runs), the scripts, and `power.log`, the Mac's sleep, wake and power-source events
  from `pmset -g log`.
- `sweep-configs/`: the labels, passed after the model's config: `long.json`, `rerun.json`, `ab.json`.
- [machine.json](machine.json): the Mac, macOS, power settings, chargers, compiler and the binary's sha256.

Local paths are replaced by placeholders: `~` for the home folder, `<repo>` for the Kapitsa checkout, `<run-root>` for
the harness's output folder, `<host>` for the host name.
