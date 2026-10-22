# M1 Pro, 16 GB: Recursive Flint 40B-A2.5B with 2-bit experts on Kapitsa and llama.cpp, 2026-10-22

Flint's 4-bit export (22.6 GB) does not fit a 16 GB Mac. unsloth's `flint-UD-IQ2_XXS.gguf`
([unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF) @ 20468edf, 11.86 GB: IQ1_S, IQ2_XXS, IQ2_S and IQ3_XXS
routed experts) does. Both engines here run that one file:

- Kapitsa's `flint:16gb`: the GGUF imported with `kapitsa import`, the routed experts kept in their IQ blocks and the
  rest requantized (attention, dense MLP and shared expert 4-bit group 64, embedding 3-bit, LM head 6-bit, router
  bf16). 11.73 GB of weights.
- llama.cpp master `d772569` (built from source with Metal, `0.7.0-dev (b1, d772569)`), the GGUF as it is.

The protocol is the poster's (`bench/harness/compare_engines.py configs/flint.json --skip agent`, labels
`flint-16gb-kapitsa` and `flint-16gb-llamacpp`), with the 4K prompt only: a 16K prompt does not fit 8K of context.
Flint reasons before every answer, so every run uses `bench.py --always-reasons`, and decode samples at the model
card's temperature 1.0 (top_p 0.95, top_k 20).

| Engine | Run | Decode T=1.0 (tok/s) | Decode greedy (tok/s) | Prefill 4K (tok/s) | Follow-up TTFT 4K (s) | Peak (GB) | Swapped during |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Kapitsa | `flint-16gb-kapitsa-20261022-184404` | 57.55 (57.45-58.36) | 57.81 | 423 | 0.18 | 11.58 | tools (3,188 pages) |
| llama.cpp d772569 | `flint-16gb-llamacpp-20261022-184806` | 42.82 (42.23-43.12) | 42.53 | 358 | 0.68 | 12.25 | decode (4,036), prefill (30,828), tools (7,452) |
| Kapitsa | `flint-16gb-kapitsa-20261022-190557` | 58.15 (57.56-58.23) | 57.79 | 422 | 0.22 | 11.58 | none |
| **Kapitsa mean** | | **57.85** | **57.80** | **422** | **0.20** | **11.58** | |

The runs are in `results/`, their server logs in `logs/`. Decode is the median of 5 runs of 512 tokens (the range in
brackets), prefill the harness's 4K prompt (3,874 of Flint's tokens), follow-up TTFT the time to the first token of a
cached second turn. Both engines answered the tool-call request.

- **Decode.** Kapitsa decodes 1.36x as fast as llama.cpp on the same file: 57.8 against 42.5 tok/s greedy. By the swap
  rule (a phase in which macOS swapped does not count) llama.cpp's greedy decode is its only clean number; its T 1.0
  decode (42.8) agrees with it.
- **Prefill.** Kapitsa reads the 4K prompt in 9.2 s (422 tok/s), llama.cpp in 10.8 s (358), which swapped 30,828
  pages (about 480 MB) during it.
- **Memory.** Kapitsa peaks at 11.58 GB, llama.cpp at 12.25 GB. Kapitsa's own budget on this Mac is 12.2 GB (the GPU's
  recommended working set, 12.7 GB, less 512 MB): the weights, one slot, 448-row prefill chunks and an 8K context fill
  11.74 GB of it.

## Quality

The import's requantization costs quality against the GGUF as llama.cpp runs it. Teacher-forced over 192 decode rows
of `logit_dump`'s windows (Kapitsa's `logit_dump` against llama.cpp's CPU backend, whose Metal backend runs out of
memory at these lengths):

| Against | KL mean | KL median | Top-1 agreement | NLL (nats/token) |
| --- | ---: | ---: | ---: | ---: |
| the GGUF as stored | 0.217 | 0.018 | 89.1% | +0.10 |
| the same weights as Kapitsa's import (written back as a GGUF) | 0.045 | 0.001 | 97.4% | +0.012 |

The second row is the engines' difference alone: both run the same weights. The first row's gap is the import's
requantization of the non-expert tensors ([#324](https://github.com/nace-ai/kapitsa/pull/324) prices each part).

## Conditions

MacBook Pro (14-inch, 2021), `MacBookPro18,3`, M1 Pro (14-core GPU), 16 GB, macOS 27.0.1 (26A434), on AC power, with
about 2.0 GB of swap in use at the start. The GPU lock (`/tmp/kapitsa-gpu.lock`) was held by the caller for each run.
No run is flagged for contention: the foreign GPU load was at most 2.1%.

**Kapitsa** ran `kapitsa serve --ctx 8192 --slots 1`, which chose 448-row prefill chunks itself; `--ctx 8192` and
`--slots 1` are what it chooses on this Mac too. The first run's binary was built from the working tree before it was
committed as `d672466` and `60caf11` (its record says `6e7a776+dirty`), the second from `60caf11`, clean. The two runs
agree within 1%. mlock held 11.43 and 11.19 GB of the weights; the rest stayed pageable, and no decode or prefill phase
swapped.

**llama.cpp** ran `llama-server` with `flint-llamacpp`'s flags (every layer on the GPU, `-fa on`, F16 KV,
`--cache-ram 0`, `-b 2048`, Flint's sampling) except `-c 8192` and `-ub 512` for a 16 GB Mac, in place of `-c 32768`
and `-ub 2048`. Other batch sizes were not run.

**Not measured:** other Macs, 16K prompts, Ollama and LM Studio (not installed on this Mac). rec and the MLX-based
engines do not run GGUF files, and Flint's MLX export (22.6 GB) does not fit here.
