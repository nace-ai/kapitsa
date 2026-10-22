# M3 Pro, 36 GB: Recursive Flint 40B-A2.5B on Kapitsa and llama.cpp, 2026-10-22

Flint 40B-A2.5B is Recursive's coding MoE (256 experts, 6 routed and a shared one a token, 2.5B active). This is the
[M4 Pro run](../2026-10-22-flint-m4pro/README.md) repeated on a MacBook Pro M3 Pro with 36 GB on macOS 14.1, with the
same configs, weights and engine builds:

- Kapitsa on Recursive's 4-bit MLX export
  ([RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4) @ 3b93f905), built from
  [#303](https://github.com/nace-ai/kapitsa/pull/303)'s `3da3798` (merged unchanged as `f206376`). Kapitsa's Flint kernels need Metal 3.2 (macOS 15)
  on main and do not compile on macOS 14. #303 runs their three last-threadgroup merges as separate dispatches there.
  **These numbers are that macOS 14 path.** On macOS 15 and later the fused kernels the M4 Pro measured run instead.
- llama.cpp b11804 (`c7fc50941`, the release's macOS arm64 build), on Recursive's Q4_1 GGUF
  ([RecursiveSI/flint-int4-gguf](https://huggingface.co/RecursiveSI/flint-int4-gguf) @ 9c471510).
- **rec 0.1.0 does not run here.** Its release binary refuses this Mac before loading anything: `✗ This Mac is not
  supported` / `rec: this model requires 48 GiB of memory; detected 36 GiB` (`logs/flint-rec-*.log`; the harness
  stored both attempts as invalid records in `results/`).

There are two measurements, as on the M4 Pro. **The poster's protocol**
(`bench/harness/compare_engines.py configs/flint.json --skip agent`) ran Kapitsa and rec in two passes in opposite
order, then llama.cpp. rec's slots in both passes ended in its refusal, within 7 s. **A context sweep** then measured
Kapitsa from 1K to 128K tokens of prompt, and llama.cpp from 1K to 64K.

Every run uses `bench.py --always-reasons`, because Flint reasons before every answer. Decode samples at the model card's
temperature 1.0 (top_p 0.95, top_k 20).

## The poster's protocol

| Pass | Engine | Run | Decode T=1.0 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Follow-up TTFT 4K / 16K (s) | Peak (GB) | Foreign GPU load |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (fwd) | Kapitsa | `flint-kapitsa-20261022-163350` | 73.96 (73.17-74.17) | 74.30 | 798 / 673 | 0.15 / 0.19 | 23.05 | 0% |
| 1 (fwd) | rec 0.1.0 | `flint-rec-20261022-163714` | × | × | × | × | – | refused: needs 48 GiB |
| 2 (rev) | rec 0.1.0 | `flint-rec-20261022-163721` | × | × | × | × | – | refused: needs 48 GiB |
| 2 (rev) | Kapitsa | `flint-kapitsa-20261022-163728` | 73.53 (73.29-74.38) | 74.64 | 800 / 675 | 0.15 / 0.17 | 23.04 | 0% |
| 3 | llama.cpp b11804 | `flint-llamacpp-20261022-164049` | 51.00 (50.67-51.48) | 51.18 | 771 / 640 | 0.37 / 0.49 | 26.56 | 0% |
| | **Kapitsa mean** | | **73.75** | **74.47** | **799 / 674** | **0.15 / 0.18** | **23.04** | |
| | **llama.cpp b11804 mean** | | **51.00** | **51.18** | **771 / 640** | **0.37 / 0.49** | **26.56** | |

The runs are in `results/` and their server logs in `logs/`. Decode is the median of 5 runs of 512 tokens, with the range
in brackets. Prefill uses the harness's 4K and 16K prompts (3,874 and 15,066 of Flint's tokens). Follow-up TTFT is the
median time to the first token of a cached second turn.

- **Prefill.** Kapitsa reads prompts 1.04x as fast as llama.cpp at 4K (799 against 771 tok/s) and 1.05x at 16K (674
  against 640).
- **Decode.** Kapitsa decodes 1.45x as fast as llama.cpp: 73.7 against 51.0 tok/s at T 1.0, and greedy 74.5 against 51.2.
- **Follow-up turns.** Kapitsa starts a cached second turn 2.4x sooner than llama.cpp at 4K (0.15 against 0.37 s) and
  2.7x sooner at 16K (0.18 against 0.49 s).
- **Memory.** Kapitsa peaks at 23.0 GB and llama.cpp at 26.6. This Mac's recommended GPU working set is 29.0 GB.
- **Against the M4 Pro.** At the same settings this Mac reads 4K prompts at 83% of the M4 Pro's speed (Kapitsa 799
  against 968) and decodes at 58% (73.7 against 127.9). llama.cpp reaches 79% and 64% (771 against 977, 51.0 against
  79.6). Kapitsa's lead over llama.cpp in decode shrinks from 1.61x to 1.45x. Part of that may be #303's extra merge
  dispatches, but they were not measured apart from everything else that differs between the two Macs (chip, memory
  bandwidth, macOS 14.1 against 27.0.1).

## Context sweep: 1K to 128K tokens

| Prefill (tok/s) | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Kapitsa** | **649** | **747** | **737** | **701** | **628** | **518** | **380** | **249** |
| llama.cpp b11804 | 523 | 781 | 768 | 723 | 636 | 506 | 350 | – |
| <sub>Kapitsa ÷ llama.cpp</sub> | <sub>1.24×</sub> | <sub>0.96×</sub> | <sub>0.96×</sub> | <sub>0.97×</sub> | <sub>0.99×</sub> | <sub>1.02×</sub> | <sub>1.08×</sub> | <sub>–</sub> |

| Decode (tok/s) | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Kapitsa** | **71.3** | **70.2** | **68.2** | **64.5** | **58.4** | **49.3** | **37.1** | **25.2** |
| llama.cpp b11804 | 51.7 | 50.8 | 49.4 | 47.1 | 44.1 | 37.9 | 27.6 | – |
| <sub>Kapitsa ÷ llama.cpp</sub> | <sub>1.38×</sub> | <sub>1.38×</sub> | <sub>1.38×</sub> | <sub>1.37×</sub> | <sub>1.32×</sub> | <sub>1.30×</sub> | <sub>1.35×</sub> | <sub>–</sub> |

- **Prefill.** From 2K to 16K, llama.cpp reads 1-4% faster than Kapitsa. The two are level at 32K (Kapitsa 1.02x) and
  Kapitsa leads by 1.08x at 64K. Kapitsa's 1.24x at 1K is a single request against llama.cpp's slowest point, so
  don't read much into it. Only Kapitsa reached 128K, at 249 tok/s, a prompt read of 8.6 minutes and 34% of its 4K speed.
- **Decode.** Kapitsa decodes 1.30-1.38x as fast as llama.cpp at every length both reached. At 128K it decodes
  25.2 tok/s, 37% of its 4K speed.
- **Kapitsa's sweep prefill is lower than its protocol prefill** (737 against 799 tok/s at 4K), while llama.cpp's
  agree (768 against 771). The sweep's `--ctx 262144` made Kapitsa choose 1024-row prefill chunks instead of 2048:
  `memory: 1024-row prefill chunks (2048 would leave less than one 262144-token conversation in the 27.8 GB budget)`
  (`context-sweep/logs/kapitsa.log`). The protocol's `--ctx 32768` keeps 2048, and the M4 Pro, which has room for
  2048 at both settings, saw its sweep match its protocol within 1%. The smaller chunks are probably part of the gap, but
  this was not measured. No sweep ran at `--ctx 32768`, and other things differ too: one cold request against the
  harness's median, a different prompt, and the battery falling from 92% to 70% during the sweep.

How it ran (`context-sweep/`): `ctx_sweep.py` (the M4 Pro's, unchanged) against each engine's server. Each server was
started by `sweep_engine.sh` (the M4 Pro's, with this Mac's paths and a `LLAMA_CTX` override) with a long context:
Kapitsa `--ctx 262144`, llama.cpp `-c 70000`. One engine ran at a time, each in its own hold of the GPU lock and warmed
first by one 2K request. The prompt, the reply and the measures are as in the
[M4 Pro sweep](../2026-10-22-flint-m4pro/README.md#context-sweep-1k-to-128k-tokens), with two differences:

- **The sizes are round thousands** (1,000 to 128,000 tokens), not the M4 Pro's powers of two (1,024 to 131,072).
  Both engines here ran the same sizes. Compared with the M4 Pro's table, each column here is 2.3% shorter.
- **llama.cpp ran with `-c 70000`, not 140000.** At `-c 140000` its F16 KV cache did not fit next to the weights.
  Every request, the warm-up's included, failed with `Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)`
  (`context-sweep/logs/llamacpp-first-attempt-c140000-oom.log`). With `-c 70000` it reaches 64K, and 128K was not run.
  Kapitsa reported room for 182,930 tokens of KV (28 KB a token) in its 27.8 GB budget, so its 128K point fits.

The source-file part of the prompt was read from this Mac's checkout of `3da3798`, 1,646,790 tokens for both engines.
Each point is a single request with no repeats, so read a point as about ±3%. The points and each server's command are
in `context-sweep/sweep.json`, and the servers' logs are in `context-sweep/logs/`.

## Conditions

MacBook Pro (16-inch, Nov 2023), `Mac15,7`, M3 Pro (12-core CPU, 18-core GPU), 36 GB, macOS 14.1 (23B2073). **The Mac
ran on battery throughout**, not on AC as the harness README recommends, with Low Power Mode off and everything under
`caffeinate -i`. The charge fell from 98% at the protocol's start to 92% at its end, and to 70% by the sweep's end.
pmset recorded no thermal or performance warning. 1.2 GB of swap was in use at the start. The GPU lock
`/tmp/kapitsa-gpu.lock` was held per engine.

**Contention.** None. The harness's sampler saw 0% foreign GPU load in every phase of every run, and no run is
flagged. The context sweep has no contention sampler.

**Kapitsa** ran one binary in every run and in the sweep (sha256 `0bd51b3833f7…`), built from `3da3798` (#303 on main
`dc5cfff`, its source clean). On this Mac #303 was checked first with `tools/logit_floor.py` and `tools/logit_compare.py`
(#303's description has the output). It ran 4-bit Flint with `--ctx 32768 --slots 1` and plain decoding.

**llama.cpp b11804** ran `llama-server` with the flags of the M4 Pro run: its model card's flags, except
`-b 2048 -ub 2048` instead of the card's `-b 256 -ub 128`, and `-c 32768 -np 1` like Kapitsa.

**Not measured:** rec (it refuses 36 GB Macs), Ollama and LM Studio (neither has Flint here), llama.cpp beyond 64K,
256K context, this Mac on AC power, and macOS 15 on this Mac, where Kapitsa would run its fused kernels.
