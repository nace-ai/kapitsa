# Results, 2026-10-18, M5 Max: Nemotron 3.5 Super-VL 120B-A12B

The result files behind the Super-VL tables of [docs/benchmarks.md](../../../docs/benchmarks.md) and
[docs/benchmarks-m5-max.md](../../../docs/benchmarks-m5-max.md): every engine that runs this model on a Mac, each at its
latest release (or, where only an unreleased build runs it, that build) with its own best settings, written by
[bench/harness](../../harness/README.md) with [configs/super-vl.json](../../harness/configs/super-vl.json) on a MacBook Pro
(`Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 25E246, AC power). Rebuild the table with

```bash
python3 bench/harness/compare_table.py bench/results/2026-10-18-m5-max --out /tmp/table
```

## Session

One engine at a time under the shared GPU lock, in four `compare_engines.py` invocations on 2026-10-18 (local time,
UTC-07:00), back to back except for swapping weights on the disk (the MLX 4-bit and GGUF files did not fit together):

| Invocation | Time | Configs |
| --- | --- | --- |
| 1 | 02:02-02:20 | `super-omlx`, `super-mlxlm` |
| 1b | 02:20-02:43 | the same two again (both runs of 1 were flagged CONTENTION) |
| 2 | 03:16-04:36 | `super-kapitsa`, `super-kapitsa-plain`, `super-mlxvlm`, `super-mlxvlm-mtp`, `super-llamacpp`, `-mtp1`, `-mtp2`, `-mtp3`, `-lilicorr` |
| 3 | 04:36-05:56 | the same nine again (all of 2 were flagged CONTENTION) |

The top-level files are the newest run of each configuration (what `compare_table.py` picks);
[confirmations/](confirmations/) holds the earlier run of each, outside the tables. [harness.log](harness.log) is the
transcript of all four invocations, [logs/](logs/) the servers' own logs, [machine.json](machine.json) the machine and
build record.

**Contention.** Every run except `super-omlx`'s second is flagged CONTENTION. The desktop stayed as it was (apps open;
nothing was closed for the session), and one `com.apple.WebKit` process used 3-5% of the GPU throughout, rising above
the 5% threshold in the prefill phases of every engine. Kapitsa's runs show much more of it (mean 11-14%, up to 64%),
always in its own prefill phases (31% mean during its 16K prefill, which runs on the GPU and the Neural Engine
together), while its decode phases saw 4-5% like everyone else's; the same pattern repeated in its second run. Kapitsa's
prefill rates here (627 / 509 tok/s) are close to its rates without a desktop load measured outside the server
(672-684 / 514-516 tok/s at 4K / 16K), so the flag marks the conditions rather than a large loss. mlx-lm's second run
also recorded swap-outs during its 16K prefill. Every configuration was run twice; the second runs (except oMLX's) were
flagged the same way, so the tables use them with the flag.

**Repetition flags.** At temperature 0.6 this model often repeats itself in the 512-token coding answer (distinct-4-gram
below 0.5) on every engine, Kapitsa included; those runs are kept and flagged, as the harness does.

## Builds and engines

| Engine | Build | How it ran |
| --- | --- | --- |
| Kapitsa | `opt/super-vl`, built from d6fb99d (the source is identical at 04a82a4, the commit the results record), Release, binary sha256 `566028726b48...` in machine.json | `serve`, 32K context, 1 slot; default (LiLiCorr drafter named with `--dflash`) and `--dflash none` |
| mlx-vlm | **unreleased**: git main 73d366e (2026-10-17), mlx 0.32.4; its result files say `version: unknown` (a fixed version string was added to the config after the session) | Super-VL support (PR #2528, with its native MTP drafter) came after the 0.7.8 release, which refuses the model |
| oMLX | 0.7.1rc1 (mlx 0.32.4, bundled mlx-lm 0.31.4.dev132) | text-only MLX 4-bit through its mlx-lm path; no speculation possible (no MTP head in those weights, no DFlash path for Nemotron-H) |
| mlx-lm | 0.32.2 (mlx 0.32.4) | the same text-only MLX 4-bit |
| llama.cpp | 0.7.0 = build b11669 (620d2225c), the latest release (it has PR #30520, Super-VL) | ggml-org's Q4_0 GGUF; plain, MTP n-max 1/2/3, LiLiCorr n-max 7 |

Not run: mlx-lm and oMLX on NVIDIA's checkpoint (`Model type nemotron_h_omni not supported`; mlx-lm's `nemotron_h` would
also not read the ModelOpt NVFP4 / per-tensor FP8 tensors), mlx-vlm 0.7.8 (same message), llama.cpp on the NVFP4 GGUF
(no Metal kernels for NVFP4). Invocation 1 ran with `configs/super-vl.json` before its commit (its oMLX and mlx-lm
entries are the committed ones; the results say `harness.dirty: true` at f0dc7f8); invocations 2 and 3 ran at the
committed 04a82a4 (one result, `super-kapitsa-plain-20261018-044423`, says dirty because a config line of another
engine was edited and reverted while it ran).

## Weights

| Engine | Repo @ revision | Files | Size |
| --- | --- | --- | ---: |
| Kapitsa, mlx-vlm | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` @ 59b8721646c38111a242da18218859fb2b454322, with `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr` @ 292bf1efc06ba39567da7351f2ad7a522afa3959 in `dflash/` | Kapitsa's store copy (`kapitsa pull super-vl`), [manifest](model-manifests/nemotron-3.5-super-vl-nvfp4.json) | 76.38 GB (75.01 GB of model safetensors) |
| oMLX, mlx-lm | `rapid-mlx/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-MLX-4bit` @ 016789035d80736393d9bfcdb63d6911002e424d | all 22 files, [manifest](model-manifests/rapid-mlx-super-vl-mlx-4bit.json) | 69.54 GB |
| llama.cpp | `ggml-org/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF` @ 3c91ad1be60865386af92e8147dafb32470b097d | `...-Q4_0.gguf` 71.80 GB, `mtp-...-Q4_0.gguf` 2.27 GB, `lilicorr-...-NVFP4.gguf` 1.35 GB, [manifest with sha256](model-manifests/ggml-org-super-vl-gguf-q4_0.json) | 75.43 GB |

Sizes are decimal GB. The MLX 4-bit and GGUF files were downloaded for the session (one at a time) and deleted after
it. The rapid-mlx repo's README says Rapid-MLX applies a "precision adapter" that stock loaders do not; oMLX and mlx-lm
read its files as they are (their answers and tool calls passed every check). The GGUF results record their model path
with the config's unexpanded placeholder; the files are the ones in the manifest.

## Smoke tests (before the session, not in the tables)

Two decode runs per setting, 4K prompt only, same harness:

- mlx-vlm MTP block size, T = 0.6 / greedy tok/s: 2: 39.5 / 41.9, 3: 44.1 / 47.2, **4: 47.4 / 50.4**, 5: 45.6 / 50.8,
  6: 39.6 / 42.7. Block 4 was used.
- llama.cpp LiLiCorr: n-max 3 (llama-server's default) 25.2 / 23.5, **n-max 7** (the drafter's full block) 30.0 / 29.4,
  plain 36.9 / 35.8; about 4.7 tokens accepted per cycle, but the drafter's NVFP4 GGUF has no Metal kernels.
- mlx-vlm with MTP fails the 16K prompt with `[METAL] Command buffer execution failed: Insufficient Memory` (peak
  108 GB) in both session runs, and also with its prefix cache off (`APC_ENABLED=0`), so its 16K cells are empty.

Local paths are replaced by placeholders: `~` for the home folder, `<repo>` for the Kapitsa checkout, `<run-root>` for
the session's output folder, `<tmp>` for the session's scratch folder. No numbers, answers, validity checks or flags
were removed.
