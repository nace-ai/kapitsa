# Results, 2026-10-18, M5 Max: Nemotron 3.5 Lightning 30B-A3B

The result files behind the Lightning tables of [docs/benchmarks.md](../../../docs/benchmarks.md) and
[docs/benchmarks-m5-max.md](../../../docs/benchmarks-m5-max.md): every engine that runs this model on a Mac, each at its
latest release (or, where only an unreleased build has the feature, that build) with its own best settings, written by
[bench/harness](../../harness/README.md) with [configs/lightning.json](../../harness/configs/lightning.json) on a
MacBook Pro (`Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 25E246, AC power). Rebuild the table with

```bash
python3 bench/harness/compare_table.py bench/results/2026-10-18-m5-max-lightning --out /tmp/table
```

## Session

One engine at a time under the shared GPU lock, in three `compare_engines.py` invocations on 2026-10-18 (local time,
UTC-07:00), back to back, at the committed 80e4608 (every result says `harness.dirty: false`), with
`--quiet-timeout 120`:

| Invocation | Time | Configs |
| --- | --- | --- |
| 1 | 08:08-08:54 | all 13: `lightning-kapitsa`, `-kapitsa-plain`, `-mlxvlm`, `-mlxvlm-dspark`, `-mlxvlm-mtp`, `-omlx`, `-omlx-mtp`, `-mlxlm`, `-llamacpp`, `-llamacpp-mtp1`, `-mtp2`, `-mtp3`, `-llamacpp-dspark` |
| 2 | 08:54-09:37 | the same 13 again (every run of 1 but oMLX + MTP's was flagged CONTENTION) |
| 3 | 09:38-09:41 | `lightning-llamacpp-mtp1` once more: its run in 2 is INVALID (see below) |

The top-level files are the newest valid run of each configuration (what `compare_table.py` picks);
[confirmations/](confirmations/) holds the other runs, outside the tables. [harness.log](harness.log) is the transcript
of all three invocations, [logs/](logs/) the servers' own logs, [machine.json](machine.json) the machine and build
record. Another agent's GPU jobs took the lock between some configurations of invocation 1 (the transcript's lock
waits); they never ran during a measurement.

**Contention.** The desktop stayed as it was (apps open; nothing was closed for the session). One
`com.apple.WebKit` GPU process used 5-7% of the GPU throughout, so the quiet wait before each run timed out (27 of 27
runs, at 120 s) and the monitor flagged the prefill phases of every run except oMLX + MTP's two and mlx-vlm's second
(foreign load mean 3.2-6.4%, at most 16%; Kapitsa's runs 5.6-6.4% mean, 10-11% at most, in its prefill phases, which
are longer than the others'). The first run of `lightning-llamacpp-mtp3` also saw Slack and the terminal app use the GPU
and swapped out during its decode phase. Every configuration was run twice; the second runs were flagged the same way,
so the tables use them with the flag.

**Invalid run.** `confirmations/lightning-llamacpp-mtp1-20261018-092413.json`: in one sampled answer the model wrote
`</think>` in its reply (it began with a sentence of planning and closed a reasoning block it had not opened), which the
harness counts as a thinking leak. Its other numbers were in line (143.1 / 144.9 tok/s decode); the configuration was
run a third time, and that run is the one in the table.

**Repetition flags.** A few 512-token answers repeat themselves (distinct-4-gram below 0.5): llama.cpp's plain greedy
run (one answer) and its MTP n-max 2 sampled run (two answers). They are kept and flagged, as the harness does.

## Builds and engines

| Engine | Build | How it ran |
| --- | --- | --- |
| Kapitsa | `opt/lightning` 80e4608 (C++ source equal to ed3ce7b), Release | `serve`, 32K context, 1 slot; default (DSpark drafter named with `--dflash`) and `--dflash none` |
| mlx-vlm | 0.7.8 (mlx 0.32.4), the latest release; and **unreleased** git main 73d366e for its MTP drafter | NVIDIA's NVFP4 checkpoint; plain, DSpark drafter (block 6), MTP (git main, block 4) |
| oMLX | 0.7.1rc1 (mlx 0.32.4, bundled mlx-lm 0.31.4.dev132) | MLX 4-bit through its mlx-lm path; plain, and Lightning MTP at fixed depth 2 with a 4-bit conversion that keeps the MTP head |
| mlx-lm | 0.32.2 (mlx 0.32.4) | the same MLX 4-bit as oMLX's plain run; no speculative path for Nemotron-H |
| llama.cpp | 0.7.0 = build b11669 (620d2225c), the latest release | ggml-org's Q4_0 GGUF; plain, MTP n-max 1/2/3, DSpark drafter GGUF n-max 3 |

Not run: mlx-lm on NVIDIA's checkpoint (`ValueError: Unsupported modelopt quant_algo: MIXED_PRECISION`: its ModelOpt
loader takes pure NVFP4 checkpoints, and this one has FP8 Mamba projections), oMLX on it (its bundled mlx-lm loads it
as a plain `nemotron_h` folder and fails: `Received 11974 parameters not in model: ...input_scale, ...weight_scale`),
oMLX with a drafter (its DFlash path covers Qwen / Gemma / GLM / Laguna / MiMo, its DSpark path DeepSeek V4), mlx-lm
with MTP (its `nemotron_h` drops the `mtp.*` weights), mlx-vlm 0.7.8 with MTP (the Nemotron-H MTP drafter came after
that release), Kapitsa with the MTP head (not implemented; Kapitsa's speculative path for Lightning is DSpark).
llama.cpp nightly b11717 was installed but not run: the latest release was the version under test.

## Weights

| Engine | Repo @ revision | Files | Size |
| --- | --- | --- | ---: |
| Kapitsa, mlx-vlm | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` @ bee7596271d1495f6992ae224aefde4410e816b8, with `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` @ 6d127b50ecd98562cea85d0087680b7d86091fb0 in `dflash/` | Kapitsa's store copy (`kapitsa pull lightning`), [manifest](model-manifests/nemotron-3.5-lightning-nvfp4.json) | 22.93 GB (21.56 GB of model safetensors, 1.35 GB of drafter) |
| oMLX, mlx-lm | `mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit` @ 55ac8c89261109b36c04371cd3f479a4594208c8 | all 12 files, [manifest](model-manifests/mlx-community-lightning-4bit.json) | 17.79 GB |
| oMLX + MTP | `EigenLabs/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit-mtp` @ 153c469494b3f16a0eaa752dae25cfc9452ce5c6 | all 13 files, [manifest](model-manifests/eigenlabs-lightning-mlx-4bit-mtp.json) | 18.54 GB |
| llama.cpp | `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` @ 8a08a1c81dadcc75d35dbb96016cfd344b632e67 | `...-Q4_0.gguf` 18.90 GB, `mtp-...-Q4_0.gguf` 1.16 GB, [manifest](model-manifests/ggml-org-lightning-gguf-q4_0.json) | 20.05 GB |
| llama.cpp + DSpark | `magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF` @ ed35fc990759330e325ab9e41bcf80737ab4dafc | `...-NVFP4-DSpark.gguf`, [manifest](model-manifests/magnitudedev-lightning-dspark-gguf.json) | 1.36 GB |

Sizes are decimal GB. Every downloaded file was checked against the Hub's sha256 (LFS files) and size. Both MLX 4-bit
conversions are MLX affine 4-bit, group 64, from NVIDIA's BF16 weights; they are not the same files: 401 of the 729
trunk arrays are byte-identical, the other 328 (the embedding, Mamba projections, shared experts, ...) were quantized
differently, and EigenLabs' adds the 34 `mtp.*` arrays oMLX's MTP path needs (mlx-community's conversion drops them).
ggml-org publishes no drafter GGUF for this model; magnitudedev's is NVIDIA's DSpark drafter converted with llama.cpp
b11669's own converter. The MLX 4-bit and GGUF files were downloaded for the session and deleted after it. The GGUF
results record their model path with the config's unexpanded placeholder; the files are the ones in the manifest.

## Smoke tests (before the session, not in the tables)

Two decode runs per setting, 4K prompt only, same harness (T = 0.6 / greedy tok/s):

- mlx-vlm 0.7.8 + DSpark drafter, block size: 4: 143.0 / 147.9, **6: 142.7 / 159.0**, 8 (the drafter's block):
  137.4 / 149.2. NVIDIA's DFlash drafter for this model (`nvidia/...-NVFP4-DFlash` @ 8abcc4db), default block:
  102.0 / 104.1. Block 6 with DSpark was used.
- mlx-vlm git main + MTP, block size: 2: 117.0 / 118.7, 3: 120.6 / 124.4, **4: 130.5 / 122.7**, 5: 118.3 / 127.0.
- oMLX Lightning MTP: fixed depth 1 (oMLX's default for Nemotron-H) 181.4 / 186.0, **fixed 2: 205.5 / 200.1**, fixed 3
  195.2 / 194.8, fixed 4 173.7 / 191.0, adaptive up to 3 179.7 / 204.5; the same folder without MTP 167.5 / 169.7.
- llama.cpp + DSpark GGUF: **n-max 3** (llama-server's default) 79.4 / 93.6, n-max 7 (the drafter's block) 67.3 / 74.6;
  2.4-2.9 tokens accepted per cycle. llama.cpp + MTP n-max 3: 166.9 / 165.3.
- Kapitsa (DSpark): 233.9 / 235.6.

Local paths are replaced by placeholders: `~` for the home folder, `<repo>` for the Kapitsa checkout, `<run-root>` for
the session's output folder, `<tmp>` for the session's scratch folder. No numbers, answers, validity checks or flags
were removed.
