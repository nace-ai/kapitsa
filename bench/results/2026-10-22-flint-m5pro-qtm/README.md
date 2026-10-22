# M5 Pro, 48 GB: Recursive Flint 40B-A2.5B on Kapitsa after #340, 2026-10-22

Kapitsa alone, rerun on the poster's Flint protocol after [#340](https://github.com/nace-ai/kapitsa/pull/340) moved
Flint's routed-expert prefill GEMMs onto the M5's tensor units. rec and llama.cpp were not rerun: their cells stay the
stored 2026-10-22 runs of [`2026-10-22-flint-m5pro`](../2026-10-22-flint-m5pro/README.md).

- Kapitsa built Release from a clean checkout of main at `91a249a` (includes #340; binary sha256 `194ce9c3d518…`), on
  Recursive's 4-bit MLX export ([RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4) @ 3b93f905).
- MacBook Pro M5 Pro, 48 GB (`Mac17,8`), macOS 26.5.2, High Power mode, AC power.
- `bench/harness/compare_engines.py configs/flint.json --only flint-kapitsa --skip agent` (bench args
  `--temperature 1.0 --always-reasons --ref-prompt-tokens 4000:3874,16000:15066`), two legs, each its own
  `lockf -k /tmp/kapitsa-gpu.lock` hold, each started only with at least 40% of memory free (89% and 87%). The driver
  and leg scripts are in [`run/`](run/). Times are this Mac's clock (UTC+5).

| Run | Decode T = 1.0 (min–max) | Greedy | Prefill 4K / 16K (tok/s) | Follow-up TTFT 4K / 16K (s) | Peak (GB) | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [`flint-kapitsa-20261022-223254`](results/flint-kapitsa-20261022-223254.json) | 151.24 (149.2–152.4) | 145.73 | 1,743 / 1,350 | 0.10 / 0.13 | 23.25 | none |
| [`flint-kapitsa-20261022-223805`](results/flint-kapitsa-20261022-223805.json) | 149.87 (149.0–151.9) | 151.14 | 2,098 / 1,423 | 0.11 / 0.14 | 23.25 | contention (16K prefill) |
| **Mean** | **150.56** | **148.44** | **1,920 / 1,387** | **0.11 / 0.13** | **23.25** | |

- **Before #340** (main `6c13588`, the same protocol, two runs): 145.74 decode, 1,374 / 1,108 prefill. The 4K prefill
  mean is 1.40x that and 16K 1.25x; decode is 3% higher.
- **The two runs' 4K prefill differs by 20%** (1,743 against 2,098). Foreign GPU load does not explain it: run 1's 4K
  phase had none, run 2's averaged 2.7% (Windows App), and load can only slow a run. The cause was not found, so read
  the mean as ±10%.
- **Contention.** Run 2 is flagged for Windows App's load during its 16K prefill (6.1% mean, 18% peak); its other
  phases are below the 5% threshold. Run 1 had 0.6% foreign load overall and no flag.
- **Against the other engines' stored cells** (prefill 4K / 16K, decode at T 1.0): rec 0.1.0 2,138 / 1,941 and 144.9,
  llama.cpp b11804 2,072 / 1,773 and 92.4. rec reads 4K prompts 1.11x as fast as Kapitsa and 16K 1.40x; llama.cpp
  1.08x and 1.28x. Kapitsa decodes 1.04x rec and 1.63x llama.cpp. rec's runs had Edge at 3-10% of the GPU and these
  had almost none, which can only favour Kapitsa here; the runs were not interleaved.

**The poster's M5 Pro Flint Kapitsa cell is the mean**: 1,920 / 151. The context sweep was not rerun; the one in
`2026-10-22-flint-m5pro` is `6c13588`'s.

Paths in the records and logs are anonymized: `<repo>` is the Kapitsa checkout, `<run-root>` the run's output folder,
`<scratch>` the driver's folder.
