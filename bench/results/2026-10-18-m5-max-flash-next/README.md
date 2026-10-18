# Results, 2026-10-18, M5 Max: Qwen3.8-Flash-Next

The result files behind the Flash-Next tables of [docs/benchmarks.md](../../../docs/benchmarks.md) and
[docs/benchmarks-m5-max.md](../../../docs/benchmarks-m5-max.md): every engine that runs this model on a Mac, each at its
latest release with its own best settings, written by [bench/harness](../../harness/README.md) with
[configs/flash-next.json](../../harness/configs/flash-next.json) on a MacBook Pro (`Mac17,6`, M5 Max, 40-core GPU,
128 GB, macOS 26.4 25E246, AC power). Rebuild the table with

```bash
python3 bench/harness/compare_table.py bench/results/2026-10-18-m5-max-flash-next --out /tmp/table
```

## Session

One engine at a time under the shared GPU lock, on 2026-10-18 (local time, UTC-07:00), `--quiet-timeout 120`, each
configuration run twice back to back. The weights of oMLX's oQ4e, llama.cpp and Ollama did not fit on the disk
together, so each competitor was downloaded, run and deleted in turn, and the session is a sequence of blocks rather
than one pass over every configuration:

| Time | Configs | Kapitsa build |
| --- | --- | --- |
| 09:55-10:54 | `flash-kapitsa`, `-kapitsa-plain` (and diagnostics, below) | 8912f1b (earlier build) |
| 10:55-11:28 | mlx-vlm smoke tests, `flash-mlxvlm`, `-mlxvlm-mtp` | |
| 11:28-11:50 | oMLX smoke tests on the pack | |
| 11:52-12:03 | `flash-kapitsa`, `-kapitsa-plain` | 62ab7ed (earlier build) |
| 12:03-12:16 | `flash-omlx`, `-omlx-mtp` | |
| 12:17-13:10 | oQ4e download, smoke tests, `flash-omlx-oq4e`, `-omlx-oq4e-mtp` | |
| 13:10-14:08 | GGUF download, smoke tests, `flash-llamacpp`, `-llamacpp-mtp4` | |
| 14:08-14:45 | Ollama MLX tag pull, `flash-ollama-mlx` | |
| 15:22-15:47 | Ollama GGUF tag pull (a first pull at 14:45 was stopped and deleted, then restarted) | |
| 15:42-16:00 | `flash-kapitsa`, `-kapitsa-plain`: the table's Kapitsa rows | 95fbcd6 (published) |
| 16:00-16:11 | `flash-ollama-gguf` | |

The top-level files are the newest valid run of each configuration (what `compare_table.py` picks);
[confirmations/](confirmations/) holds the first runs, outside the tables. [harness.log](harness.log) is the transcript
of every invocation (smoke tests and diagnostics included), [logs/](logs/) the servers' own logs,
[machine.json](machine.json) the machine record. Other agents' GPU jobs (Kapitsa development) took the lock between
some configurations; they never ran during a measurement (the transcript's lock waits). A Hugging Face upload of the
pack ran in the background during the morning blocks, and the Ollama GGUF download was still writing to disk during
the first `flash-kapitsa` run of 95fbcd6 (15:42-15:47; the download ended at 15:46:53): that run, in
[confirmations/](confirmations/), wrote 98.9 / 102.6 tok/s against the table's 101.8 / 107.6.

**Contention.** The desktop stayed as it was. A `com.apple.WebKit` GPU process used 3-6% of the GPU most of the day,
more during prefill, so the quiet wait timed out in most runs and the monitor flagged the prefill phases of most runs.
Worse: during mlx-vlm's second pass (11:18-11:28) and both oQ4e passes (12:50-13:10) a Google Chrome helper used the GPU
heavily (mean 7-10% for mlx-vlm, 14-16% for oQ4e, peaks of 30-60%), and every phase of those runs is flagged. The oQ4e
rows are probably low: their smoke tests just before (2 runs, 4K) gave 110.7 / 105.6 tok/s for MTP depth 2, against
105.5 / 98.7 in the table. oMLX's runs on the pack were the cleanest (unflagged but for one prefill phase). Swap stayed
at 16-20 GB used all day.

**Invalid and failed.** Every 4K and 16K prompt of `flash-ollama-mlx` fails (both runs): Ollama's MLX runner holds
about 98-100 GiB with this tag, and the prefill pushes it past the 107.5 GiB Metal working-set limit ("Insufficient
Memory"); its decode and tool-call results are valid. The second run of `flash-llamacpp-mtp4` fails one tool-call check
(streaming, thinking off: the model called `run_shell` to list the workspace instead of `read_file`); its first run
passed every check. A few 512-token answers repeat themselves (distinct-4-gram below 0.5) and are kept and flagged.

## Earlier Kapitsa builds

[earlier-builds/](earlier-builds/) holds the Kapitsa runs of the builds measured before the published one, outside the
tables:

- **8912f1b** (opt/flash-next before the fix): its server crashed on every 16K prompt (4 of 4 runs, both with and
  without MTP: a Metal "Invalid Resource" command-buffer error), which also ended each run before the tool-call phase.
  `flash-kapitsa-4k` / `-plain-4k` are the same servers without the 16K prompt (`--sizes 4000`), so their tool calls
  ran (all passed). [diagnostics/](earlier-builds/8912f1b/diagnostics/) holds the runs that narrowed it down (4K then
  8K or 16K prompts crash; a 16K prompt alone once ended in "n-gram row out of range"; `NI_QSA_PF=list` and
  `--prefill-chunk 16384` did not avoid it reliably). Cause: the sparse-attention scratch buffers grown during a long
  prompt were released while earlier command buffers of the same prompt still read them; fixed in 62ab7ed.
- **62ab7ed** (the fix; MTP at depth 1, the default then, `--spec 1`): two runs of each Kapitsa configuration, full protocol, all valid: MTP 104.0 / 102.1 tok/s
  sampled, 103.2 / 103.1 greedy; plain 83.7 / 85.2 and 86.1 / 86.8; prefill 858-898 tok/s at 4K, 869-902 at 16K;
  follow-ups 0.08 s; 70.5 GB.

## Builds and engines

| Engine | Build | How it ran |
| --- | --- | --- |
| Kapitsa | `opt/flash-next` 95fbcd6, Release (binary sha256 8e7766f0...; MTP depth 2 by default from this commit) | `serve`, 32K context, 1 slot; its default MTP (depth 2) and `--spec 0` |
| oMLX | 0.7.1rc1 (mlx 0.32.4), the latest release | the pack (its vendored mlx-vlm qwen4_exp model) plain and with its Lightning MTP at fixed depth 3; Jundot's oQ4e plain and with MTP at fixed depth 2 |
| mlx-vlm | 0.7.8 (mlx 0.32.4), the latest release (git main 73d366e adds nothing for qwen4_exp) | the pack, plain and with its qwen4_exp MTP drafter, block 4 |
| llama.cpp | 0.7.0 = build b11669 (620d2225c), the latest release | ggml-org's IQ4_NL GGUF, plain and MTP n-max 4 |
| Ollama | 0.40.6, the latest release | its MLX runner on the official `125b-mlx` (NVFP4) tag with built-in adaptive MTP; its GGUF runner on the official `125b-a6b-q4_K_M` tag (no MTP draft in that tag) |

Not run: mlx-lm (no release has `qwen4_exp`; its support is the open PR #1788, which also drops the MTP head; an
unreleased PR was not built). LM Studio and MTPLX were not measured. Kapitsa with a block drafter: none exists for this model.

## Weights

| Engine | Repo @ revision | Files | Size |
| --- | --- | --- | ---: |
| Kapitsa, mlx-vlm, oMLX | `nace-ai/Qwen3.8-Flash-Next-MLX-4bit` @ 5d423f9f563c1f1b7ca7945b268e65c0b92b0eae | all 66 files, [manifest](model-manifests/nace-ai-qwen3.8-flash-next-mlx-4bit.json) | 105.86 GB |
| mlx-vlm + MTP | the pack's own MTP head as a drafter folder (same tensors) | [manifest](model-manifests/qwen3.8-flash-next-4bit-mlxvlm-mtp-drafter.json) | 1.51 GB |
| oMLX (oQ4e) | `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` @ c4ddfb54a967a9bc74d7da16425f55f0872f8aac | all 34 files, [manifest](model-manifests/jundot-qwen3.8-flash-next-oq4e-mtp.json) | 106.32 GB |
| llama.cpp | `ggml-org/Qwen3.8-Flash-Next-GGUF` @ 0a61d2b8d851e0bb9f22b973befd714413d63c6f | `...-IQ4_NL-0000{1,2}-of-00002.gguf` 102.05 GB, `mtp-...-Q4_0.gguf` 2.20 GB, [manifest](model-manifests/ggml-org-qwen3.8-flash-next-gguf-iq4_nl.json) | 104.25 GB |
| Ollama MLX | `qwen3.8-flash-next:125b-mlx` (= `125b-a6b-nvfp4`), manifest sha256:c7b96ffe21af | 1,658 tensors, [manifest](model-manifests/ollama-qwen3.8-flash-next-125b-mlx.json) | 104.85 GB |
| Ollama GGUF | `qwen3.8-flash-next:125b-a6b-q4_K_M`, manifest sha256:f79436a1cc83 | Q4_K_M GGUF 119.15 GB + vision projector 0.91 GB, [manifest](model-manifests/ollama-qwen3.8-flash-next-125b-a6b-q4_K_M.json) | 120.06 GB |

Sizes are decimal GB. Every downloaded file was checked against the Hub's sha256 (LFS files) and size; Ollama checks
each blob's digest itself. The pack the runs read is Kapitsa's converter output, checked against the published
revision (all 66 files identical). The competitors' weights were downloaded for the session and deleted after their
runs.

The recipes, side by side (the n-gram table and the routed experts are 98% of the parameters):

| | Nace pack | oQ4e (Jundot, oMLX's quantizer) | ggml-org IQ4_NL | Ollama NVFP4 (MLX tag) |
| --- | --- | --- | --- | --- |
| Routed experts | 4-bit g64 | 4-bit g64 | IQ4_NL (4.5 bits/weight) | NVFP4 |
| n-gram table | 4-bit g32, own file, memory-mapped by Kapitsa and mlx-vlm, resident in oMLX | 4-bit g32, resident in oMLX | Q4_0, memory-mapped by llama.cpp | NVFP4 |
| Shared experts | 4-bit g64 | 8-bit g128 | Q8_0 | |
| Gated DeltaNet projections | 4-bit g64 | in_proj_qkv 4-bit g64 (26 layers) or 5-6-bit; in_proj_z/a/b 5-bit g128 or 6-bit g64; out_proj 5-bit g128 | Q8_0 | |
| Attention, indexer | 6-bit g64 (attention), 8-bit g64 (indexer) | 5-8-bit g64 | Q8_0 (attention), BF16 (indexer) | |
| Hyper-connections | 6-bit g64 (inject weights BF16) | 5/6/8-bit g64 | Q8_0 before attention, IQ4_NL before the MoE | |
| Embedding, LM head | 8-bit g64, 4-bit g64 | 8-bit g64, 8-bit g64 | Q8_0, Q8_0 | |
| Routers | 8-bit g64 | BF16 | F32 | |
| MTP head | in the pack (1.51 GB; experts 4-bit, other linears 8-bit) | in the checkpoint (1.48 GB) | separate Q4_0 GGUF | in the tag |
| Calibration | no importance matrix; per group the best of min-max, symmetric and least-squares fits | an importance matrix (1024 samples x 512 tokens, `oq_imatrix_report.json` in the repo) and a weighted least-squares refit of scale and bias | not recorded | not recorded |
| Size | 105.86 GB (4.7 bits/weight) | 106.32 GB (4.72 bits/weight) | 102.05 + 2.20 GB | 104.85 GB |

oQ4e's mix varies by layer (its quantizer allocates bits by measured sensitivity); the counts are in its
`config.json`; the GGUF's types are read from its tensor headers. Ollama's tag lists its tensors without their formats (blank cells).

## Smoke tests (before the runs, not in the tables)

Two decode runs per setting, 4K prompt only, same harness (T = 0.6 / greedy tok/s):

- mlx-vlm 0.7.8 + MTP: the drafter's default (configured block 2, adaptive) 74.5 / 82.4, block 2 66.2 / 67.1,
  3 77.4 / 83.5, **4 80.4 / 87.5**, 5 77.5 / 79.4.
- oMLX + MTP on the pack: adaptive (default, up to 3) 94.3 / 91.7, fixed 1 86.6 / 91.9, 2 102.7 / 104.8,
  **3 109.7 / 111.3**, 4 101.3 / 99.7, 5 95.2 / 101.0; without MTP 56.5 / 56.6.
- oMLX + MTP on oQ4e: adaptive 108.2 / 97.6, **fixed 2 110.7 / 105.6**, 3 104.7 / 101.2, 4 106.0 / 96.0.
- llama.cpp + MTP: n-max 1 65.7 / 67.1, 2 82.6 / 86.0, 3 85.5 / 91.1, **4 94.7 / 97.3**, 5 89.0 / 100.9,
  6 83.7 / 96.2.

Local paths are replaced by placeholders: `~` for the home folder, `<repo>` for the Kapitsa checkout the harness ran
from (8912f1b; its runner code equals the published build's), `<repo-62ab7ed>` / `<repo-95fbcd6>` for the
checkouts of those Kapitsa builds, `<run-root>` / `<smoke-root>` / `<diag-root>` for the session's output folders,
`<tmp>` for its scratch folder. No numbers, answers, validity checks or flags were removed.
