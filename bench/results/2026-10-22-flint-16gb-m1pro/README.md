# M1 Pro, 16 GB: Recursive Flint 40B-A2.5B with 2-bit experts on Kapitsa and llama.cpp, 2026-10-22

Flint's 4-bit export (22.6 GB) does not fit a 16 GB Mac. unsloth's `flint-UD-IQ2_XXS.gguf`
([unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF) @ 20468edf, 11.86 GB: IQ1_S, IQ2_XXS, IQ2_S and IQ3_XXS
routed experts) does. Both engines here run that one file:

- **Kapitsa's `flint:16gb`.** The GGUF is imported with `kapitsa import`. The routed experts stay in their IQ blocks
  and the rest is requantized: attention, the dense MLP and the shared expert to 4-bit group 64, the embedding to 3-bit,
  the LM head to 6-bit, with the router kept in bf16. 11.73 GB of weights.
- **llama.cpp** master `d772569` (built from source with Metal, `0.7.0-dev (b1, d772569)`), on the GGUF as it is.

The protocol is the poster's (`bench/harness/compare_engines.py configs/flint.json --skip agent`, labels
`flint-16gb-kapitsa` and `flint-16gb-llamacpp`), with the 4K prompt only: a 16K prompt does not fit 8K of context.
Flint reasons before every answer, so every run uses `bench.py --always-reasons`, and decode samples at the model
card's temperature 1.0 (top_p 0.95, top_k 20).

## Alternated with llama.cpp (the final build)

Four runs one after another, 22:48-23:03, llama.cpp and Kapitsa in turn (L K L K), each in its own hold of the GPU
lock. Kapitsa is `2b5ab35`: this PR's head after the rebase onto main and the review fixes (its decode path is the
`47960e8` runs' below). The Mac held 1.4-3.0 GB of swap at the start of each run and swapped during some phase of every
run: llama.cpp 12K-129K pages a phase, Kapitsa 2K-16K.

| Order | Engine | Run | Decode T=1.0 (tok/s) | Decode greedy (tok/s) | Prefill 4K (tok/s) | Follow-up TTFT 4K (s) | Peak (GB) | Swapped during (pages) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| L | llama.cpp d772569 | `flint-16gb-llamacpp-20261022-224836` | 42.22 (41.91-43.39) | **42.87** | 355 | 0.55 | 11.89 | decode 12,476, prefill 128,756 |
| K | Kapitsa `2b5ab35` | `flint-16gb-kapitsa-20261022-225312` | 60.89 (58.25-61.80) | 61.06 | 414 | 0.21 | 11.58 | decode 2,500, greedy 2,412, prefill 15,716, tools 7,668 |
| L | llama.cpp d772569 | `flint-16gb-llamacpp-20261022-225553` | 42.65 (41.96-43.26) | 42.10 | 346 | 2.23 | 11.90 | decode 32,992, greedy 21,528, prefill 71,696 |
| K | Kapitsa `2b5ab35` | `flint-16gb-kapitsa-20261022-230039` | **61.09** (60.49-61.74) | 61.09 | 413 | 0.20 | 11.58 | greedy 8,788, prefill 6,384, tools 7,356 |

Bold: the decode phases without swap.

- **Decode.** Kapitsa decodes 1.44x as fast as llama.cpp in both modes: 61.0 against 42.4 tok/s at T 1.0 (the means)
  and 61.1 against 42.5 greedy. Counting only phases without swap, it is 1.43x: Kapitsa 61.09 (T 1.0, its second run)
  against llama.cpp 42.87 (greedy, its first).
- **Prefill.** Kapitsa reads the 4K prompt in 9.4 s (413-414 tok/s), llama.cpp in 10.9-11.2 s (346-355). Every prefill
  phase swapped, llama.cpp's 72K-129K pages, Kapitsa's 6K-16K.
- **Memory.** Kapitsa peaks at 11.58 GB, llama.cpp at 11.9 GB here (12.25 GB in the earlier run). Kapitsa's own budget
  on this Mac is 12.2 GB (the GPU's recommended working set, 12.7 GB, less 512 MB): the weights, one slot, 448-row
  prefill chunks and an 8K context fill 11.74 GB of it.

## Earlier runs

The same protocol earlier the same day, not alternated with llama.cpp: two runs of the branch before its rebase onto
main, llama.cpp's first run between them, then two runs of the rebased branch (`47960e8`).

| Build | Engine | Run | Decode T=1.0 (tok/s) | Decode greedy (tok/s) | Prefill 4K (tok/s) | Follow-up TTFT 4K (s) | Peak (GB) | Swapped during (pages) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| before the rebase | Kapitsa | `flint-16gb-kapitsa-20261022-184404` | 57.55 (57.45-58.36) | 57.81 | 423 | 0.18 | 11.58 | tools 3,188 |
| | llama.cpp d772569 | `flint-16gb-llamacpp-20261022-184806` | 42.82 (42.23-43.12) | 42.53 | 358 | 0.68 | 12.25 | decode 4,036, prefill 30,828, tools 7,452 |
| before the rebase | Kapitsa | `flint-16gb-kapitsa-20261022-190557` | 58.15 (57.56-58.23) | 57.79 | 422 | 0.22 | 11.58 | none |
| rebased, `47960e8` | Kapitsa | `flint-16gb-kapitsa-20261022-200059` | 60.99 (60.78-61.95) | 61.17 | 413 | 0.21 | 11.58 | prefill 9,156, tools 6,368 |
| rebased, `47960e8` | Kapitsa | `flint-16gb-kapitsa-20261022-200335` | 61.61 (61.29-62.18) | 61.43 | 414 | 0.29 | 11.58 | tools 820 |

The runs are in `results/`, their server logs in `logs/`. Decode is the median of 5 runs of 512 tokens (the range in
brackets), prefill the harness's 4K prompt (3,874 of Flint's tokens), follow-up TTFT the time to the first token of a
cached second turn. Both engines answered the tool-call request in every run. The pair closest in time, 18:44 and
18:48, reads 1.34x (57.55 against 42.82 at T 1.0) for the build before the rebase; the rebase made decode 4% faster
(below).

## The two builds

The first two Kapitsa runs are the branch before it was rebased onto main. One ran on the uncommitted tree (its record
says `6e7a776+dirty`), the other on its commit `60caf11`.

Main had meanwhile gained [#303](https://github.com/nace-ai/kapitsa/pull/303), which runs Flint's decode merges in a
dispatch of their own on macOS 14. The branch's fix for the M1 (Apple7), whose last threadgroup read stale data
behind the fences, now uses that path: `split_merges()` before Apple9. It replaced the branch's own workaround: the
routing in a dispatch of its own, and decode attention on the verify rows' three dispatches.

The rebased build decodes faster:

- **`kapitsa serve`, 512 tokens at T 1.0, three answers per server, interleaved:** before 58.9-59.3 tok/s, rebased
  61.3-61.6 (A B A B, 12 answers).
- **The harness, in the same session:** the pre-rebase binary ran once more after the two rebased runs and read 58.3
  (greedy 58.6), with prefill 412.

That last run is not stored. Its record would name this checkout's commit, not its binary's.

Prefill did not change with the rebase: the logits of `logit_dump`'s prefill rows are bitwise identical across the two
builds, and the pre-rebase binary's last run read 412. The 413-414 tok/s of the later runs, against 422-423 earlier,
is this Mac's state over the session.

## Quality

The import's requantization costs quality against the GGUF as llama.cpp runs it. These are teacher-forced over 192
decode rows of `logit_dump`'s windows: Kapitsa's `logit_dump` against llama.cpp's CPU backend, whose Metal backend runs
out of memory at these lengths. They are measured on the pre-rebase build.

| Against | KL mean | KL median | Top-1 agreement | NLL (nats/token) |
| --- | ---: | ---: | ---: | ---: |
| the GGUF as stored | 0.217 | 0.018 | 89.1% | +0.10 |
| the same weights as Kapitsa's import (written back as a GGUF) | 0.045 | 0.001 | 97.4% | +0.012 |

The second row is the engines' difference alone: both run the same weights. The first row's gap is the import's
requantization of the non-expert tensors ([#324](https://github.com/nace-ai/kapitsa/pull/324) prices each part).

The rebased build's decode rows, against the pre-rebase build's:
- KL median 3e-7, mean 0.013;
- top-1 99%, with 2 flips after a near-tie of margin 0.015;
- NLL 0.006 nats lower.

They are bitwise the same in two runs.

## Conditions

MacBook Pro (14-inch, 2021), `MacBookPro18,3`, M1 Pro (14-core GPU), 16 GB, macOS 27.0.1 (26A434), on AC power, with
1.4-3.0 GB of swap in use at the start of each run. The GPU lock (`/tmp/kapitsa-gpu.lock`) was held by the caller for
each run. No run is flagged for contention: the foreign GPU load was at most 2.3%.

**Kapitsa** ran `kapitsa serve --ctx 8192 --slots 1`, which chose 448-row prefill chunks itself; `--ctx 8192` and
`--slots 1` are what it chooses on this Mac too. mlock held 11.2-11.4 GB of the weights; the rest stayed pageable, and
the Mac still swapped: three of Kapitsa's four decode phases (T 1.0 and greedy in each alternated run) swapped 2K-9K pages; only the
second run's T 1.0 decode is swap-free (the bold 61.09 above), as the table lists.

**llama.cpp** ran `llama-server` with `flint-llamacpp`'s flags, except `-c 8192` and `-ub 512` (in place of `-c 32768`
and `-ub 2048`) for a 16 GB Mac. Those flags put every layer on the GPU and set `-fa on`, F16 KV, `--cache-ram 0`,
`-b 2048` and Flint's sampling. Other batch sizes were not run.

**Not measured:** other Macs, 16K prompts, Ollama and LM Studio (not installed on this Mac). rec and the MLX-based
engines do not run GGUF files, and Flint's MLX export (22.6 GB) does not fit here.
