<picture>
  <source media="(max-width: 600px)" srcset="docs/assets/header-narrow.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/header-dark.svg">
  <img alt="Kapitsa: a blazing fast Apple Silicon optimized inference engine built for Qwen and Nemotron. Writing speed of Qwen3.6 35B-A3B on a MacBook Pro (2024), M4 Pro, 48 GB, in tokens per second, at 4-bit: Kapitsa with KFlash 161, Splash with DFlash2 135, oMLX with DFlash2 127, Ollama with MTP 89, LM Studio 86, MTPLX with MTP 80, llama.cpp with MTP 69." src="docs/assets/header-light.svg">
</picture>

Kapitsa is an inference engine for Apple silicon that runs Qwen and Nemotron models locally, built for coding agents. It starts opencode, Claude Code, Codex or pi on a local model, and serves OpenAI and Anthropic APIs.

[Quickstart](#quickstart) · [CLI](#cli) · [Benchmarks](#benchmarks) · [Models](#models) · [How KFlash works](docs/kflash.md) · [Docs](docs/usage.md)

## 🔥 What's new

- [2026/10] 🚀 **KFlash: tree speculative decoding, up to 3.5× llama.cpp and nearly 2× oMLX.** One verify pass checks a whole tree of the drafter's guesses ([how it works](docs/kflash.md)).
- [2026/10] 🏆 **Kapitsa: the fastest decode in 16 of 17 model and Mac pairs** against eight engines, up to 2.3× MLX ([poster](#benchmarks)).
- [2026/10] 🤝 **KFlash budget: subagents decode together.** Three requests at once share one verify pass, up to **+23% total throughput** on the 27B.
- [2026/10] ⚡ **QMT prefill: read the prompt up to 1.5× faster on M5.** Metal 4 matrix units for the prompt (4B, M5 Pro).
- [2026/10] 💾 **SSD prefix cache: restart the server, keep the conversation.** A 16K-token follow-up on the 35B starts in **0.16 s instead of 17 s**, about 100× (`--disk-cache`).
- [2026/10] 🧠 **Prefix reuse: subagents start warm.** A new question over the shared system prompt answers in **0.06 s instead of 1.9 s** (4K prompt, 4B, M5 Pro).
- [2026/10] 🧪 **Output check: speed has to prove it kept the answers.** A default changes only after its logits stay within twice the model's noise floor ([how](CONTRIBUTING.md#output-check)).
- [2026/10] 📊 **`kapitsa bench compare`: see it on your own Mac.** The same test against every engine installed here, in one command.

## Quickstart

**Requirements:** a Mac with Apple silicon (M1 or newer) and macOS 14 Sonoma or later.

```bash
brew install nace-ai/tap/kapitsa
# or: curl -fsSL https://raw.githubusercontent.com/nace-ai/kapitsa/main/install.sh | bash
```

```bash
kapitsa opencode --model 35b    # also: claude, codex, pi
```

<sub>On a 16-24 GB Mac use `--model 9b`, or `4b` for the quickest replies.</sub>

## CLI

```bash
kapitsa serve                   # chat + OpenAI/Anthropic APIs at http://127.0.0.1:8000
kapitsa claude                  # Claude Code on Kapitsa
kapitsa doctor                  # this Mac's GPU and memory, and which models fit it
kapitsa bench                   # benchmark this Mac, share it with the community
kapitsa bench compare           # the same test against the other engines installed here
kapitsa pull 27b                # download a model without starting it
kapitsa list                    # downloaded models and their size
kapitsa rm 27b                  # delete a downloaded model
kapitsa import model.gguf       # use a Qwen GGUF from LM Studio, Ollama or llama.cpp
kapitsa update                  # update now (it also updates itself)
```

<sub>`--model` picks the model and its precision (`35b:6bit`). `kapitsa <command> --help` has the options.
[Usage guide](docs/usage.md)</sub>

## Benchmarks

Kapitsa has the fastest decode in 16 of the 17 model and Mac pairs measured against eight engines ([how each number was measured](docs/benchmarks.md#how-the-readme-poster-was-measured)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmarks-dark.webp">
  <img alt="Poster of prefill / decode speed for Kapitsa and eight other engines on a MacBook Pro M5 Pro, M4 Pro and M1 Pro; the numbers are in the poster" src="docs/assets/benchmarks-light.webp">
</picture>

<sub>**Each cell reads prefill / decode in tokens per second.** M5 Max results are being re-measured. Prefill is a 4K-token prompt, decode a 512-token answer. × cannot run that model · — not measured. How these were measured, run dates and flags: [docs/benchmarks.md](docs/benchmarks.md#how-the-readme-poster-was-measured).</sub>

<details>
<summary>Models not on the poster (decode tok/s)</summary>

🟩 fastest per model · — not measured

### MacBook Pro (16-inch, 2024) · M4 Pro · 48 GB

#### Nemotron

| Model | <mark>**Kapitsa**</mark> | <sub>oMLX<br>0.7.1rc1</sub> | <sub>llama.cpp<br>b11669</sub> | <sub>MLX<br>mlx-lm 0.32.2,<br>mlx-vlm 0.7.8</sub> |
| --- | ---: | ---: | ---: | ---: |
| Nemotron Dory1.0 4B<br><sub>NVFP4</sub> | 🟩&nbsp;<mark>**64&nbsp;tok/s**</mark><br><sub>loops</sub> | —<br><sub>no support</sub> | 6<br><sub>no spec</sub> | 48<br><sub>no spec</sub> |

### MacBook Pro (14-inch, 2021) · M1 Pro · 16 GB

#### Nemotron

| Model | <mark>**Kapitsa**</mark> | <sub>oMLX<br>0.7.1rc1</sub> | <sub>llama.cpp<br>b11669</sub> | <sub>MLX<br>mlx-lm 0.32.2,<br>mlx-vlm 0.7.8</sub> |
| --- | ---: | ---: | ---: | ---: |
| Nemotron Dory1.0 4B<br><sub>NVFP4</sub> | 🟩&nbsp;<mark>**40&nbsp;tok/s**</mark><br><sub>no spec</sub> | —<br><sub>no support</sub> | 4<br><sub>no spec</sub> | 30<br><sub>no spec</sub> |
| Nemotron 3 Nano 4B<br><sub>FP8</sub> | 🟩&nbsp;<mark>**37&nbsp;tok/s**</mark><br><sub>prompt lookup</sub> | —<br><sub>not run</sub> | 28<br><sub>Q4_K_M</sub> | 21<br><sub>BF16</sub> |
| Nemotron Nano 9B v2<br><sub>NVFP4</sub> | 🟩&nbsp;<mark>**25&nbsp;tok/s**</mark><br><sub>prompt lookup</sub> | —<br><sub>not run</sub> | 14<br><sub>Q4_K_M</sub> | 18 |

### MacBook Pro (16-inch, 2026) · M5 Pro · 48 GB

#### Nemotron

| Model | <mark>**Kapitsa**</mark> | <sub>oMLX<br>0.7.1rc1</sub> | <sub>mlx‑serve<br>26.10.2</sub> | <sub>Ollama<br>0.40.6</sub> | <sub>MTPLX<br>2.12.2</sub> | <sub>LM&nbsp;Studio<br>MLX runtime 1.13.1,<br>llama.cpp runtime 2.60.0</sub> | <sub>llama.cpp<br>b11669</sub> | <sub>MLX<br>mlx-lm 0.32.2,<br>mlx-vlm 0.7.8</sub> |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nemotron Dory1.0 4B<br><sub>NVFP4</sub> | 🟩&nbsp;<mark>**70&nbsp;tok/s**</mark><br><sub>loops</sub> | —<br><sub>no support</sub> | —<br><sub>no support</sub> | —<br><sub>no support</sub> | —<br><sub>no support</sub> | —<br><sub>no support</sub> | 13 | 53 |
| Nemotron 3.5 Bolt 74B‑A9B<br><sub>2/4/8-bit mix</sub> | 🟩&nbsp;<mark>**79&nbsp;tok/s**</mark><br><sub>LiLiCorr</sub> | 39<br><sub>no spec</sub> | —<br><sub>no support</sub> | —<br><sub>no support</sub> | —<br><sub>no support</sub> | —<br><sub>no support</sub> | —<br><sub>no Metal support</sub> | 41 |

</details>

Every recorded run, with its Mac, model, prefill and decode speed, is in the [benchmark table](bench/README.md). Measure your Mac: `kapitsa bench --submit` adds it to the [community results](bench/community/RESULTS.md). Another engine faster on your Mac? `kapitsa bench compare` ranks every engine you have installed, and `--report` writes a Markdown report.

## Models

<details>
<summary>The 11 models, their downloads and decoding</summary>

| Model | Download | Decoding |
| --- | --- | --- |
| **Qwen3.5-4B** · `4b`<br><sub>dense</sub> | **4-bit · 4.6 GB**<br><sub>also 6-bit 5.6 · 8-bit 6.7 · bf16 10.6 GB</sub> | **KFlash**<br><sub>z-lab's DFlash drafter; or MTP</sub> |
| **Qwen3.5-9B** · `9b`<br><sub>dense</sub> | **4-bit · 9.0 GB** | **KFlash**<br><sub>z-lab's DFlash drafter; or MTP</sub> |
| **Qwen3.8-27B** · `27b`<br><sub>dense</sub> | **4-bit · 20.2 GB**<br><sub>also 5-bit 23.5 · 6-bit 26.9 · 8-bit 33.6 · bf16 58.8 GB</sub> | **KFlash**<br><sub>DFlash2 drafter; or MTP</sub> |
| **Qwen3.6-35B-A3B** · `35b`<br><sub>MoE, 3B active</sub> | **4-bit · 22.1 GB**<br><sub>also 5-bit 26.4 · 6-bit 30.7 · 8-bit 39.4 · bf16 71.9 GB; 16-24 GB Macs: `16gb`, 2-bit experts (unsloth's GGUF) 10.8 GB</sub> | **KFlash**<br><sub>DFlash2 drafter; or MTP; `16gb`: none</sub> |
| **Qwen3.8-Flash-Next 180B-A7B** · `flash-next`<br><sub>MoE, 7B active; 128 GB Macs</sub> | **4-bit · 105.9 GB**<br><sub>4/6/8-bit MLX; its 32 GB n-gram table stays on the SSD</sub> | MTP |
| **Nemotron Dory1.0 4B** · `dory`<br><sub>looped dense, 1-4 loops per token; text only</sub> | **NVIDIA NVFP4 · 3.2 GB**<br><sub>also bf16 8.0 GB</sub> | — |
| **Nemotron 3 Nano 4B** · `nano-4b`<br><sub>hybrid Mamba-2, dense; text only</sub> | **NVIDIA FP8 · 5.3 GB**<br><sub>also bf16 8.0 GB</sub> | prompt lookup |
| **Nemotron Nano 9B v2** · `nano-9b`<br><sub>hybrid Mamba-2, dense; text only</sub> | **NVIDIA NVFP4 · 7.9 GB** | prompt lookup |
| **Nemotron 3.5 Lightning 30B-A3B** · `lightning`<br><sub>hybrid Mamba-2 MoE, 3B active; text only</sub> | **NVIDIA NVFP4 · 22.9 GB**<br><sub>4-bit NVFP4 experts, FP8, BF16</sub> | **KFlash**<br><sub>NVIDIA's DSpark drafter</sub> |
| **Nemotron 3.5 Bolt 74B-A9B** · `bolt`<br><sub>hybrid Mamba-2 MoE, 9B active</sub> | **NVIDIA 24 GB · 25.1 GB**<br><sub>2-bit experts, NVFP4, 8-bit</sub> | **KFlash**<br><sub>NVIDIA's LiLiCorr drafter</sub> |
| **Nemotron 3.5 Super-VL 120B-A12B** · `super`<br><sub>hybrid Mamba-2 MoE, 12B active; 96 GB+ Macs</sub> | **NVIDIA NVFP4 · 76.4 GB**<br><sub>4-bit NVFP4 experts, FP8, BF16</sub> | **KFlash**<br><sub>NVIDIA's LiLiCorr drafter</sub> |

<sub>Other precisions: `--model 35b:6bit` is five times closer to bf16 than the 4-bit default ([KL](docs/benchmarks.md#quality-qwen36-35b-a3b-against-bf16-2026-10-19)). All but Lightning, Dory, the two Nemotron Nanos and the 35B's `16gb` read images. Memory: 16 GB for the 4B and 9B models, 32 GB
for the 27B, 35B, Lightning and Bolt, 96 GB for Super-VL, 128 GB for Flash-Next. On a 16 or 24 GB Mac, `35b` takes its `16gb` variant: unsloth's 2-bit GGUF, downloaded and imported in one step ([how](docs/usage.md#gguf-files)). [Details](docs/usage.md#models-and-downloads)</sub>

</details>

## How it works

<picture>
  <source media="(max-width: 600px)" srcset="docs/assets/how-it-works-narrow.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/how-it-works-dark.svg">
  <img alt="How Kapitsa works: DFlash2 vs KFlash. DFlash2, the drafter's chain, as Splash, oMLX and mlx-serve run it: the drafter guesses the next 7 tokens, the big model checks them in one pass and keeps them up to the first wrong guess (return a, 2 tokens). KFlash, Kapitsa's: in the same one pass the big model checks a best-first tree of the drafter's top candidates in 8 rows; where the drafter is unsure its runner-ups ride along, and the longest branch the big model agrees with is kept (return a + b, 4 tokens). Same output as plain decoding, tree size measured on each Mac, 8-17% more tokens per pass on short prompts. Below: GPU, hand-written Metal code; Neural Engine, helps read long prompts on the 4B, 27B, Bolt and Super-VL; Memory, keeps conversations ready, up to 4 at once." src="docs/assets/how-it-works-light.svg">
</picture>

Kapitsa decodes with **KFlash**. A draft model such as DFlash2 guesses the next seven tokens, with a few candidates for each one. Engines that run it as published (Splash, oMLX, mlx-serve) check one chain of guesses and stop at the first wrong one. KFlash checks a best-first tree of the drafter's candidates in the same pass, so where the drafter hesitates, its runner-ups are checked too and more tokens survive each pass. The tree's size comes from what a verify row costs on your Mac, measured there, and it changes no output: the big model checks every token. It runs on the published drafters as they are: z-lab's DFlash, DFlash2, and NVIDIA's DSpark and LiLiCorr. [How KFlash works](docs/kflash.md)

## Get release notices

<p align="center"><img src="docs/assets/watch-releases.gif" alt="On the Kapitsa repository page: Watch, then Custom, tick Releases, Apply" width="720"></p>

<sub>Watch → Custom → Releases on [nace-ai/kapitsa](https://github.com/nace-ai/kapitsa) puts every new version in your notifications. A drawn illustration of the steps, not a screen recording; the page source is in [tools/readme_images](tools/readme_images/README.md).</sub>

<sub>[How KFlash works](docs/kflash.md) · [Architecture](docs/architecture.md) · [Contributing](CONTRIBUTING.md) · [Press kit](docs/assets/brand/README.md) ·
Apache 2.0, © 2026 [Nace AI](https://nace.ai) · [Third-party notices](THIRD_PARTY_NOTICES)</sub>
